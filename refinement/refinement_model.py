import re
from typing import Optional
from utils.logging_config import get_logger, log_isabelle_interaction

logger = get_logger(__name__)

_MAX_CONSECUTIVE_REGRESSIONS = 2
_STAGNATION_THRESHOLD = 2


def _format_axiom_block(name: str, formula: str) -> str:
    idx = name.split('_')[-1]
    return f'(* Generated Premise {idx} *)\naxiomatization where\n  {name}: "{formula}"'


def _inject_axiom_into_using(code: str, goal_to_name: dict) -> str:
    """Add auto-asserted axiom names to the `using` clause of matching have steps.

    For each (goal, axiom_name) pair, finds `have step_N: "goal"` and appends
    axiom_name to its `using ...` line so Sledgehammer can use it directly.
    """
    for goal, axiom_name in goal_to_name.items():
        # Match: have <label>: "<formula>" possibly across lines, followed by using
        pattern = re.compile(
            r'(have\s+\w+:\s+"' + re.escape(goal) + r'"\s*\n\s*using\s+)([^\n]+)',
            re.MULTILINE
        )
        def _add_name(m, aname=axiom_name):
            using_line = m.group(2)
            if aname not in using_line:
                using_line = using_line.rstrip() + ' ' + aname
            return m.group(1) + using_line
        code = pattern.sub(_add_name, code)
    return code


class RefinementModel():
    def __init__(self, generative_model, critique_model,
                 prompt_dict: Optional[dict] = None,
                 auto_assert: bool = True):
        self.generative_model = generative_model
        self.critique_model = critique_model
        self.prompt_dict = prompt_dict
        self.auto_assert = auto_assert
        self.max_premises = 10  # Total axiom cap across all iterations
        self.max_new_per_step = 3  # Max new axioms added in a single refinement step
        self.current_isabelle_code = None  # Tracks Isabelle theory across iterations

    def _refine_axioms_formal(self, isabelle_code: str,
                               critique_output: dict,
                               failed_attempt: Optional[str] = None,
                               frozen_names: set = None) -> str:
        """Refine generated premise axioms directly in Isabelle code.

        Uses the LLM to modify/add/remove generated_premise_* axioms
        based on unsolved goals and error feedback from Isabelle.

        Args:
            failed_attempt: If provided, axiom text from a previous refinement
                that made things worse. Included as a negative example in the
                prompt so the LLM tries a different approach.
            frozen_names: Set of generated_premise_* names that appeared in
                successful proof tactics and must not be dropped.
        """
        formaliser = self.critique_model.formaliser
        frozen_names = frozen_names or set()

        # Extract current generated axioms
        current_axioms = formaliser.extract_generated_axioms(isabelle_code)
        current_count = len(current_axioms)

        # Format unsolved goals
        unsolved_goals = critique_output.get('unsolved_goals', [])
        unsolved_goals_text = ''
        if unsolved_goals:
            unsolved_goals_text = '\n'.join(
                f"Goal {i+1}: {g}" for i, g in enumerate(unsolved_goals)
            )

        # Format axiom names
        axiom_names_text = '\n'.join(
            a['name'] for a in current_axioms
        ) if current_axioms else '(none)'

        # Format failed attempt feedback
        failed_attempt_info = ''
        if failed_attempt:
            failed_attempt_info = (
                "IMPORTANT: The following previous refinement attempt made "
                "the proof WORSE (more unsolved goals). Do NOT repeat these "
                "axioms — try a fundamentally different approach:\n"
                f"{failed_attempt}"
            )

        log_isabelle_interaction(
            direction="isabelle_to_llm",
            content=f"Unsolved goals: {unsolved_goals}",
            context={"stage": "formal_axiom_refinement"}
        )

        predicate_defs_text = formaliser._build_predicate_defs_text()

        # Call LLM for formal refinement
        refined_output = self.generative_model.generate(
            model_prompt_dir='refinement_model',
            prompt_name=self.prompt_dict['refine axioms'],
            isabelle_code=isabelle_code,
            unsolved_goals=unsolved_goals_text,
            generated_axiom_names=axiom_names_text,
            predicate_definitions=predicate_defs_text,
            max_new=str(self.max_new_per_step),
            max_total=str(self.max_premises),
            failed_attempt_info=failed_attempt_info
        )

        logger.debug(f"LLM refined axiom output:\n{refined_output}")

        # Splice refined axioms into the code
        updated_code = formaliser.replace_generated_axioms(
            isabelle_code, refined_output
        )

        # Inject explicitly typed consts declared by the LLM (preserves correct
        # types and definitions). add_consts_if_needed runs after as a fallback
        # for any predicates not explicitly declared.
        explicit_consts, _, explicit_defs = formaliser._parse_axiom_code(refined_output)
        formaliser.const_definitions.update(explicit_defs)
        if explicit_consts:
            existing_names = {
                m.group(1)
                for m in re.finditer(r'^\s+(\w+)\s*::', updated_code, re.MULTILINE)
            }
            fresh = {k: v for k, v in explicit_consts.items() if k not in existing_names}
            if fresh:
                new_lines = '\n'.join(f'  {n} :: "{t}"' for n, t in fresh.items())
                updated_code = re.sub(
                    r'(consts\b.*?\n)((?:  \S.*\n)*)',
                    lambda m: m.group(0) + new_lines + '\n',
                    updated_code, count=1, flags=re.DOTALL
                )
                logger.info(f"Injected explicitly typed consts from refinement: {list(fresh.keys())}")

        # Add any remaining new consts inferred from formula usage (fallback)
        updated_code = formaliser.add_consts_if_needed(
            updated_code, refined_output
        )

        # Re-insert / restore frozen premises.
        # The LLM must not drop OR change the formula of a frozen axiom.
        if frozen_names:
            old_axiom_map = {a['name']: a for a in current_axioms}
            new_axiom_map = {a['name']: a for a in formaliser.extract_generated_axioms(updated_code)}
            needs_rebuild = False
            for name in sorted(frozen_names):
                if name not in old_axiom_map:
                    continue
                locked_formula = old_axiom_map[name]['formula']
                if name not in new_axiom_map:
                    logger.info(f"Restoring frozen premise dropped by LLM: {name}")
                    needs_rebuild = True
                elif new_axiom_map[name]['formula'] != locked_formula:
                    logger.info(f"Frozen premise {name} formula changed by LLM — restoring original")
                    needs_rebuild = True
            if needs_rebuild:
                # Force-override every frozen axiom back to its locked formula
                all_new = {a['name']: a for a in formaliser.extract_generated_axioms(updated_code)}
                for name in sorted(frozen_names):
                    if name in old_axiom_map:
                        all_new[name] = old_axiom_map[name]
                ordered = sorted(all_new.values(), key=lambda a: a['name'])
                combined = '\n\n'.join(_format_axiom_block(a['name'], a['formula']) for a in ordered)
                updated_code = formaliser.replace_generated_axioms(isabelle_code, combined)
                updated_code = formaliser.add_consts_if_needed(updated_code, refined_output)

        # Enforce axiom count cap
        new_axioms = formaliser.extract_generated_axioms(updated_code)
        max_allowed = min(current_count + self.max_new_per_step, self.max_premises)
        if len(new_axioms) > max_allowed:
            logger.info(f"Capping axioms from {len(new_axioms)} to {max_allowed}")
            # Keep only the first max_allowed axioms
            # Re-extract the axiom blocks text for the kept ones
            kept_blocks = [_format_axiom_block(a['name'], a['formula'])
                           for a in new_axioms[:max_allowed]]
            kept_text = '\n\n'.join(kept_blocks)
            updated_code = formaliser.replace_generated_axioms(
                updated_code, kept_text
            )

        # Ensure all refined axioms have explicit quantifiers
        final_axioms = formaliser.extract_generated_axioms(updated_code)
        needs_rebuild = False
        for a in final_axioms:
            fixed = formaliser._add_missing_quantifiers(a['formula'])
            if fixed != a['formula']:
                needs_rebuild = True
                a['formula'] = fixed
        if needs_rebuild:
            rebuilt_blocks = [_format_axiom_block(a['name'], a['formula'])
                              for a in final_axioms]
            rebuilt_text = '\n\n'.join(rebuilt_blocks)
            updated_code = formaliser.replace_generated_axioms(
                updated_code, rebuilt_text
            )
            logger.info("Added missing quantifiers to refined axioms")

        logger.info(f"Refined axioms ({len(final_axioms)}): "
                     f"{[a['name'] for a in final_axioms]}")

        return updated_code

    def refine(self, hypothesis: list, premise: Optional[list] = None,
               generated_premises: Optional[list] = None,
               data_name: str = 'example',
               iterations: int = 8) -> dict:

        # if premise is empty, set it to None
        self.current_isabelle_code = None  # Reset for new refinement

        history_generated_premises = []
        history_critique_output = []
        result = {}
        self._bridge_attempts = []       # Bridge axiom strings tried so far (for retry context)
        self._bridge_new_consts = []     # New const names introduced by bridge axioms

        # Best-state tracking for regression prevention
        best_isabelle_code = None
        best_critique_output = None
        best_unsolved_count = float('inf')
        best_iteration = -1
        consecutive_regressions = 0
        max_consecutive_regressions = _MAX_CONSECUTIVE_REGRESSIONS
        last_failed_refinement = None
        best_tactic_hints = {}  # Solved tactics from best iteration, used as hints
        proof_regeneration_attempted = False  # One-shot flag for metavar rebinding
        auto_asserted_names = set()   # Names of axioms injected via direct goal assertion
        auto_asserted_goals = set()   # Goal strings already auto-asserted (gate for re-firing)
        stagnant_iterations = 0  # Consecutive iterations with no reduction in unsolved count
        frozen_premise_names = set()  # Premises referenced in successful proof tactics

        # Allow up to one extra iteration beyond `iterations` in case a
        # `continue` on the final scheduled iteration generates new code
        # (bridge injection / proof regeneration) that was never evaluated.
        max_iterations = iterations + 1
        for i in range(max_iterations):
            logger.info(f"Iteration {i}")
            logger.debug(f"Premise: {premise}")
            logger.debug(f"Hypothesis: {hypothesis}")

            if i == 0:
                # Iteration 0: full NL → formalisation pipeline
                logger.info("Starting NL formalisation (iteration 0)")
                to_append = f'{i} iteration: NL generated_premises={generated_premises}'
                history_generated_premises.append(to_append)

                critique_output = self.critique_model.critique(
                    iteration_number=i,
                    explanation=generated_premises or [],
                    hypothesis=hypothesis,
                    premise=premise
                )
                # Capture the Isabelle code as source of truth
                self.current_isabelle_code = critique_output.get('code', '')
            else:
                # Iteration 1+: formal-only path (skip formalisation)
                logger.info("Starting formal-only critique (iteration > 0)")
                current_axioms = self.critique_model.formaliser.extract_generated_axioms(
                    self.current_isabelle_code
                )
                to_append = (f'{i} iteration: axioms='
                             f'{[a["name"] for a in current_axioms]}')
                history_generated_premises.append(to_append)

                critique_output = self.critique_model.critique(
                    iteration_number=i,
                    explanation=[],
                    hypothesis=hypothesis,
                    premise=premise,
                    isabelle_code=self.current_isabelle_code,
                    tactic_hints=best_tactic_hints
                )
                # Update code (syntax fixes may have changed it)
                self.current_isabelle_code = critique_output.get(
                    'code', self.current_isabelle_code
                )

            history_critique_output.append(f'{i} iteration: {critique_output}')
            logger.info("Critique results received")

            # Track best state for regression prevention
            if critique_output.get('syntactic validity', False):
                if critique_output.get('semantic validity', False):
                    unsolved_count = 0
                else:
                    unsolved_count = len(critique_output.get('unsolved_goals', []))
                    # If no goals were extracted but proof still failed, there is
                    # at least one implicit unsolved goal (e.g. an obtain/sorry step
                    # not tracked in unsolved_goals). Treat as 1 so this state is
                    # saved as best and regression detection works correctly.
                    if unsolved_count == 0 and not critique_output.get('semantic validity', False):
                        unsolved_count = 1

                if unsolved_count < best_unsolved_count:
                    best_unsolved_count = unsolved_count
                    best_isabelle_code = self.current_isabelle_code
                    best_critique_output = critique_output
                    best_iteration = i
                    consecutive_regressions = 0
                    stagnant_iterations = 0
                    last_failed_refinement = None
                    best_tactic_hints = critique_output.get('solved_tactics', {})
                    logger.info(f"New best: {best_unsolved_count} unsolved goal(s) at iteration {i}")
                else:
                    stagnant_iterations += 1

            if not critique_output['syntactic validity']:
                if i == 0 and not proof_regeneration_attempted:
                    # At iteration 0 only: iteratively rebind metavar types
                    # until Isabelle accepts the proof structure.
                    proof_regeneration_attempted = True
                    theory_name_0 = f'{self.critique_model.theory_name}_{i}'
                    rebound_code = self.critique_model.validate_and_rebind(
                        theory_name_0, self.current_isabelle_code, max_attempts=3
                    )
                    # Only retry if rebinding actually produced a different
                    # (hopefully fixed) theory; give up immediately if it did not.
                    if rebound_code != self.current_isabelle_code:
                        self.current_isabelle_code = rebound_code
                        logger.info("Iterated type-check at iter 0 — retrying")
                        continue
                    logger.warning("validate_and_rebind made no changes — giving up")
                logger.error("Syntactic error — giving up")
                result = self._build_result(
                    semantic_validity=critique_output['semantic validity'],
                    isabelle_code=self.current_isabelle_code,
                    refined_iteration=None,
                    premise=premise, hypothesis=hypothesis,
                    auto_asserted_names=auto_asserted_names,
                    history_generated_premises=history_generated_premises,
                    history_critique_output=history_critique_output,
                    unsolved_goals=critique_output.get('unsolved_goals', []),
                )
                result['refined generated premises'] = None
                break

            if critique_output['semantic validity']:
                logger.info("Generated premises are logically valid - proof found!")
                # Convert refined axioms to natural language for evaluation
                nl_premises = self._axioms_to_nl(
                    self.current_isabelle_code, hypothesis
                )
                nl_bridge = self._axioms_to_nl(
                    self.current_isabelle_code, hypothesis, bridge=True
                )
                result = self._build_result(
                    semantic_validity=critique_output['semantic validity'],
                    isabelle_code=self.current_isabelle_code,
                    refined_iteration=i,
                    premise=premise, hypothesis=hypothesis,
                    auto_asserted_names=auto_asserted_names,
                    history_generated_premises=history_generated_premises,
                    history_critique_output=history_critique_output,
                    nl_generated_premises=nl_premises,
                    bridge_axioms=self._axioms_to_list(self.current_isabelle_code, bridge=True),
                    nl_bridge_axioms=nl_bridge,
                )
                break
            else:
                # Proof failed — refine
                logger.warning("No proof found - starting refinement")

                # Check if bridge axioms are needed
                # (Isar: show step failed after all have steps proved;
                #  Legacy: apply (rule scheme) failed)
                bridge_needed = critique_output.get('bridge_needed', False)
                apply_failed = critique_output.get('apply_failed', False)

                # Try abductive instantiation if using an argumentation scheme.
                # Retried every iteration bridge_needed is True (like refinement),
                # with prior attempts passed as context so the model avoids repeating them.
                has_scheme = getattr(self.critique_model.formaliser, 'current_scheme', None) is not None
                logger.debug(
                    f"Bridge axiom check: has_scheme={has_scheme}, "
                    f"bridge_needed={bridge_needed}, apply_failed={apply_failed}, "
                    f"prior_attempts={len(self._bridge_attempts)}"
                )

                if has_scheme and (bridge_needed or apply_failed):
                    scheme_name = (getattr(self.critique_model.formaliser, 'current_scheme', None) or {}).get('name', 'unknown')
                    logger.info(
                        f"Attempting bridge axioms for scheme: {scheme_name}"
                        + (f" (retry #{len(self._bridge_attempts)})" if self._bridge_attempts else "")
                    )

                    bridge_axioms = self.critique_model.generate_bridge_axioms(
                        isabelle_code=critique_output.get('code', ''),
                        previous_attempts=self._bridge_attempts
                    )

                    if bridge_axioms:
                        self._bridge_attempts.append(bridge_axioms)
                        logger.info("Generated bridge axioms, injecting into theory")
                        theory_name = f'{self.critique_model.theory_name}_{i}'
                        augmented_code, new_consts = self.critique_model.inject_bridge_axioms(
                            critique_output['code'], bridge_axioms, theory_name
                        )
                        self._bridge_new_consts.extend(new_consts)
                        self.current_isabelle_code = augmented_code
                        logger.info("Bridge axioms injected, continuing with refinement this iteration")
                    else:
                        logger.debug("No bridge axioms generated")

                # Auto-assert remaining unsolved goals directly as axioms
                # once the proof has stagnated.
                current_unsolved = critique_output.get('unsolved_goals', [])
                new_unasserted = [g for g in current_unsolved if g not in auto_asserted_goals]
                if (self.auto_assert
                        and stagnant_iterations >= _STAGNATION_THRESHOLD
                        and new_unasserted):
                    formaliser = self.critique_model.formaliser
                    current_axioms = formaliser.extract_generated_axioms(
                        self.current_isabelle_code)
                    next_idx = len(current_axioms) + 1
                    new_blocks = []
                    for goal in new_unasserted:
                        if next_idx <= self.max_premises:
                            name = f'generated_premise_{next_idx}'
                            new_blocks.append(_format_axiom_block(name, goal))
                            auto_asserted_names.add(name)
                            auto_asserted_goals.add(goal)
                            next_idx += 1
                    if new_blocks:
                        existing_blocks = '\n\n'.join(
                            _format_axiom_block(a['name'], a['formula'])
                            for a in current_axioms
                        )
                        combined = (existing_blocks + '\n\n' + '\n\n'.join(new_blocks)
                                    if existing_blocks else '\n\n'.join(new_blocks))
                        self.current_isabelle_code = formaliser.replace_generated_axioms(
                            self.current_isabelle_code, combined
                        )
                        # Patch the using clause of have steps whose formula
                        # matches an auto-asserted goal, so Sledgehammer sees
                        # the new axiom when trying to close that step.
                        first_idx = next_idx - len(new_blocks)
                        goal_to_name = {
                            goal: f'generated_premise_{first_idx + k}'
                            for k, goal in enumerate(new_unasserted[:len(new_blocks)])
                        }
                        self.current_isabelle_code = _inject_axiom_into_using(
                            self.current_isabelle_code, goal_to_name)
                        # Freeze auto-asserted axioms so the LLM cannot drop them
                        frozen_premise_names.update(auto_asserted_names)
                        logger.info(
                            f"Auto-asserted {len(new_blocks)} unsolved goal(s) "
                            f"as axioms: {auto_asserted_names}"
                        )
                        continue

                # Check for regression before refining (scheme path only —
                # in no_scheme the proof structure changes freely so a higher
                # subgoal count doesn't mean things got worse)
                if (has_scheme
                        and unsolved_count > best_unsolved_count
                        and best_isabelle_code is not None):
                    consecutive_regressions += 1
                    logger.warning(
                        f"Regression detected "
                        f"(attempt {consecutive_regressions}/{max_consecutive_regressions}): "
                        f"{unsolved_count} unsolved vs best "
                        f"{best_unsolved_count} at iteration {best_iteration}"
                    )

                    if consecutive_regressions >= max_consecutive_regressions:
                        logger.warning(
                            "Max consecutive regressions reached, "
                            "returning best result"
                        )
                        result = self._build_result(
                            semantic_validity=best_critique_output['semantic validity'],
                            isabelle_code=best_isabelle_code,
                            refined_iteration=best_iteration,
                            premise=premise, hypothesis=hypothesis,
                            auto_asserted_names=auto_asserted_names,
                            history_generated_premises=history_generated_premises,
                            history_critique_output=history_critique_output,
                            unsolved_goals=best_critique_output.get('unsolved_goals', []),
                        )
                        break

                    # Save failed axioms as negative example, rollback
                    last_failed_refinement = '\n'.join(self._axioms_to_list(
                        self.current_isabelle_code))
                    self.current_isabelle_code = best_isabelle_code
                    logger.info("Rolled back to best code for re-refinement")

                # Formal axiom refinement (all iterations)
                # Use best critique output if we rolled back (its unsolved
                # goals match the rolled-back code)
                refinement_critique = (best_critique_output
                                       if last_failed_refinement
                                       and best_critique_output
                                       else critique_output)

                # Update frozen premises: accumulate any generated_premise_*
                # names that appear in successful proof tactics across all
                # iterations, so the LLM cannot drop them on the next pass.
                for tactic_str in critique_output.get('proof tactics', []):
                    frozen_premise_names.update(
                        re.findall(r'generated_premise_\d+', tactic_str)
                    )
                if frozen_premise_names:
                    logger.debug(f"Frozen premise names: {frozen_premise_names}")

                # Skip refinement when the only remaining issue is the show
                # step — bridge axioms are the right fix, not axiom rewrites.
                # In the no_scheme path there are no have-step goals to track,
                # so always refine when the proof hasn't succeeded.
                show_only = has_scheme and (
                    not current_unsolved
                    or current_unsolved == ['?thesis (show step)']
                )
                if show_only:
                    logger.info(
                        "Only show step remaining — skipping refinement, "
                        "relying on bridge axioms"
                    )
                else:
                    logger.info("Refining axioms in formal logic")
                    self.current_isabelle_code = self._refine_axioms_formal(
                        self.current_isabelle_code, refinement_critique,
                        failed_attempt=last_failed_refinement,
                        frozen_names=frozen_premise_names
                    )

                if i == max_iterations - 1:
                    logger.error(f'Generated premises not valid after {iterations} iterations')
                    # Use best result if available
                    best_code = (best_isabelle_code
                                 if best_isabelle_code else
                                 self.current_isabelle_code)
                    best_output = (best_critique_output
                                   if best_critique_output else
                                   critique_output)
                    result = self._build_result(
                        semantic_validity=best_output['semantic validity'],
                        isabelle_code=best_code,
                        refined_iteration=best_iteration if best_iteration >= 0 else None,
                        premise=premise, hypothesis=hypothesis,
                        auto_asserted_names=auto_asserted_names,
                        history_generated_premises=history_generated_premises,
                        history_critique_output=history_critique_output,
                        unsolved_goals=best_output.get('unsolved_goals', []),
                    )

            logger.debug(f"Completed iteration {i}")

        # Fallback: if `continue` was called on the last iteration (e.g. after
        # proof regeneration or bridge injection), the loop exits without setting
        # `result`. Return the best state seen so far.
        if not result:
            best_code = best_isabelle_code or self.current_isabelle_code
            best_output = best_critique_output
            result = self._build_result(
                semantic_validity=best_output.get('semantic validity', False) if best_output else False,
                isabelle_code=best_code,
                refined_iteration=best_iteration if best_iteration >= 0 else None,
                premise=premise, hypothesis=hypothesis,
                auto_asserted_names=auto_asserted_names,
                history_generated_premises=history_generated_premises,
                history_critique_output=history_critique_output,
                unsolved_goals=best_output.get('unsolved_goals', []) if best_output else [],
            )
            logger.warning("result was empty after loop (continue on last iteration) — using best available state")

        logger.info(f"Refinement complete: semantic_validity={result.get('semantic validity', 'N/A')}")
        return result

    def _axioms_to_list(self, isabelle_code: str, bridge: bool = False) -> list:
        """Extract axiom formulas as a list of strings."""
        formaliser = self.critique_model.formaliser
        extract = formaliser.extract_bridge_axioms if bridge else formaliser.extract_generated_axioms
        axioms = extract(isabelle_code)
        return [a['formula'] for a in axioms]

    def _axioms_to_nl(self, isabelle_code: str, hypothesis: str, bridge: bool = False) -> list:
        """Convert axiom formulas to a list of natural language sentences."""
        formaliser = self.critique_model.formaliser
        extract = formaliser.extract_bridge_axioms if bridge else formaliser.extract_generated_axioms
        axioms = extract(isabelle_code)
        if not axioms:
            return []

        axiom_formulas = '\n'.join(
            f'{i+1}. {a["name"]}: {a["formula"]}'
            for i, a in enumerate(axioms)
        )

        predicate_defs_text = formaliser._build_predicate_defs_text()

        nl_output = self.generative_model.generate(
            model_prompt_dir='refinement_model',
            prompt_name=self.prompt_dict['convert to nl'],
            hypothesis=hypothesis,
            predicate_definitions=predicate_defs_text,
            axiom_formulas=axiom_formulas
        )

        kind = 'bridge' if bridge else 'generated'
        logger.info(f"{kind.capitalize()} axioms converted to NL: {nl_output}")
        lines = [re.sub(r'^\d+\.\s*', '', l).strip()
                 for l in nl_output.strip().splitlines() if l.strip()]
        return lines

    def _build_result(self, *, semantic_validity, isabelle_code,
                      refined_iteration, premise, hypothesis,
                      auto_asserted_names, history_generated_premises,
                      history_critique_output, **extra) -> dict:
        result = {
            'semantic validity': semantic_validity,
            'premise': premise,
            'hypothesis': hypothesis,
            'refined generated premises': self._axioms_to_list(isabelle_code) if isabelle_code else None,
            'refined_isabelle_code': isabelle_code,
            'refined iteration': refined_iteration,
            'auto_asserted_axioms': sorted(auto_asserted_names),
            'bridge_new_consts': sorted(set(self._bridge_new_consts)),
            'metavar_bindings': getattr(self.critique_model.formaliser, '_last_metavar_bindings', {}),
            'history generated premises': history_generated_premises,
            'history critique output': history_critique_output,
        }
        result.update(extra)
        return result
