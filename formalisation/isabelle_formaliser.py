from .abstract import FormalisationModel
from sympy import symbols, Implies, Not, And, Or, Equivalent
from sympy.logic.inference import satisfiable
import sympy
from typing import Optional
from itertools import product
import re
import yaml
from utils.logging_config import get_logger, log_llm_prompt, log_llm_response

logger = get_logger(__name__)


class IsabelleFormaliser(FormalisationModel):

    def __init__(self, llm, prompt_dict: Optional[dict] = None):
        super().__init__(llm, prompt_dict)
        self.llm = llm
        self.logical_form = None
        self.logical_proposition = ''
        self.prompt_dict = prompt_dict
        self.isabelle_dir = self._get_isabelle_dir()
        self._pending_axiom_names = []
        self._last_metavar_bindings = {}  # Bindings from most recent _get_metavar_bindings call
        # Caches for fixed sentences (premise/hypothesis) across refinement iterations
        self._parsing_cache = {}
        self._davidsonian_cache = {}
        # Accumulated definitions for consts introduced during refinement/bridge
        self.const_definitions = {}  # name -> definition string

    def _get_isabelle_dir(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config['isabelle']['master_dir']

    def _add_quotes(self, isabelle_code: str) -> str:
        assumes_pattern = r'(assumes asm: )(.*)'
        shows_pattern = r'(shows )(.*)'

        def add_quotes_to_line(match):
            content = match.group(2)
            if not content.startswith('"') and not content.endswith('"'):
                content = f'"{content}"'
            return f'{match.group(1)}{content}'
        isabelle_code = re.sub(assumes_pattern,
                               add_quotes_to_line, isabelle_code)
        isabelle_code = re.sub(shows_pattern,
                               add_quotes_to_line, isabelle_code)
        return isabelle_code

    def _fix_c_style_application(self, isabelle_code: str) -> str:
        """Convert C-style function application Pred(x, y) to Isabelle-style Pred x y.

        Only operates inside quoted strings to avoid touching outer syntax.
        Also converts Not(x) / Not x to \\<not> x.
        """
        def fix_quoted(m):
            s = m.group(1)
            # Convert Not(...) / Not followed by word to \<not>
            s = re.sub(r'\bNot\s*\(([^)]+)\)', lambda mm: f'\\<not> {mm.group(1).strip()}', s)
            s = re.sub(r'\bNot\s+(\S)', r'\\<not> \1', s)
            # Iteratively flatten Pred(a, b, ...) -> Pred a b ...
            # Stop when no more parenthesized applications remain
            for _ in range(10):
                new_s = re.sub(
                    r'(\b[A-Z][A-Za-z0-9_]*)\(([^()]+)\)',
                    lambda mm: mm.group(1) + ' ' + ' '.join(
                        a.strip() for a in mm.group(2).split(',')
                    ),
                    s
                )
                if new_s == s:
                    break
                s = new_s
            return f'"{s}"'

        return re.sub(r'"([^"]*)"', fix_quoted, isabelle_code)

    def _fix_unicode_symbols(self, isabelle_code: str) -> str:
        """Replace Unicode logic symbols with Isabelle ASCII equivalents."""
        replacements = {
            '→': '-->',
            '∧': '\\<and>',
            '∨': '\\<or>',
            '∀': '\\<forall>',
            '∃': '\\<exists>',
            '¬': '\\<not>',
            '↔': '<-->',
        }
        for unicode_sym, isabelle_sym in replacements.items():
            isabelle_code = isabelle_code.replace(unicode_sym, isabelle_sym)
        return isabelle_code

    def _fix_assume_quantifier(self, isabelle_code: str) -> str:
        def replace_quantifier(match):
            quantifier_str = match.group(1)
            # Match \<forall> or \<exists> followed by variables and a dot
            new_quantifier_str = re.sub(r'\\<(?:forall|exists)>.*?\.\s*', '', quantifier_str)
            return f'assumes asm: "{new_quantifier_str}"'

        assumes_pattern = r'assumes asm: "(.*?)"'
        isabelle_code = re.sub(assumes_pattern, replace_quantifier,
                               isabelle_code)

        return isabelle_code

    def _clean_proof(self, isabelle_code: str) -> str:
        pattern = r'(proof -).*?(qed)(?!.*qed)'
        return re.sub(pattern, r'\1  \n  \n  \n\2',
                      isabelle_code, flags=re.DOTALL)

    def _remove_brackets(self, isabelle_code: str) -> str:
        assumes_pattern = r'(assumes asm: ")(.+)(")'
        shows_pattern = r'(shows ")(.+)(")'
        assumes_match = re.search(assumes_pattern, isabelle_code)
        if assumes_match:
            assumes_content = assumes_match.group(2)
            if '(' in assumes_content and ')' in assumes_content:
                assumes_content = re.sub(r'[\(\),]', ' ', assumes_content)
                isabelle_code = isabelle_code[:assumes_match.start(2)] + assumes_content + isabelle_code[assumes_match.end(2):]
        shows_match = re.search(shows_pattern, isabelle_code)
        if shows_match:
            shows_content = shows_match.group(2)
            if '(' in shows_content and ')' in shows_content:
                shows_content = re.sub(r'[\(\),]', ' ', shows_content)
                isabelle_code = isabelle_code[:shows_match.start(2)] + shows_content + isabelle_code[shows_match.end(2):]
        return isabelle_code

    # ==================== Helper Methods for Sentence-by-Sentence Processing ====================

    def _split_sentences(self, text: str) -> list:
        """Split text into individual sentences."""
        if not text or text.strip() == '':
            return []
        sentences = [s.strip() for s in re.split(r'[.\n]\s*', text) if s.strip()]
        return sentences

    def _aggregate_parsing_results(self, results: list) -> str:
        """Aggregate individual parsing results into combined format."""
        sections = {
            "Hypothesis Sentence": [],
            "Generated Premise": [],
            "Existing Premise": []
        }
        for sentence_type, order, result in results:
            sections[sentence_type].append((order, result))

        output_parts = []
        for section_name in ["Hypothesis Sentence", "Generated Premise", "Existing Premise"]:
            items = sections[section_name]
            if items:
                section_output = f"{section_name}:\n"
                for order, result in sorted(items, key=lambda x: x[0]):
                    # Clean up the result to just get the parsing
                    cleaned_result = re.sub(
                        r'^.*?(answer:)?\s*', '', result,
                        flags=re.DOTALL | re.IGNORECASE
                    ).strip()
                    if cleaned_result:
                        section_output += f"{cleaned_result}\n"
                output_parts.append(section_output)

        return "\n".join(output_parts)

    def _aggregate_davidsonian_results(self, results: list) -> str:
        """Aggregate individual Davidsonian form results."""
        sections = {
            "Hypothesis Sentence": [],
            "Generated Premise": [],
            "Existing Premise": []
        }
        for sentence_type, order, result in results:
            sections[sentence_type].append((order, result))

        output_parts = []
        for section_name in ["Hypothesis Sentence", "Generated Premise", "Existing Premise"]:
            items = sections[section_name]
            if items:
                section_output = f"{section_name}:\n"
                for order, result in sorted(items, key=lambda x: x[0]):
                    # Extract the relevant content from the result
                    cleaned = self._extract_davidsonian_content(result, section_name)
                    if cleaned:
                        # Strip any existing numbered prefix, then prepend the correct order
                        # so _extract_logical_forms_by_section can reliably parse it
                        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
                        section_output += f"{order}. {cleaned}\n\n"
                output_parts.append(section_output.strip())

        # Add premise sentence section if missing
        combined = "\n\n".join(output_parts)
        if 'premise sentence' not in combined.lower() and 'existing premise' not in combined.lower():
            combined += '\n\nPremise Sentence:\nLogical form: none'

        return combined

    def _extract_davidsonian_content(self, result: str, section_type: str) -> str:
        """Extract relevant content from a single Davidsonian form result."""
        # Remove "Answer:" prefix if present
        result = re.sub(r'^.*?answer:\s*', '', result, flags=re.DOTALL | re.IGNORECASE)

        # Filter to keep only relevant lines
        lines = result.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if line starts with a number followed by a dot (e.g., "1.", "10.", "123.")
            is_numbered = bool(re.match(r'^\d+\.', line))
            if is_numbered or line.lower().startswith(('subject', 'verb', 'direct', 'linking',
                                         'logical form', 'adverbial', 'prepositional',
                                         'subject complement', 'auxiliary', 'main verb',
                                         '-', 'object', 'predicates')):
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _parse_logical_result(self, result: str) -> tuple:
        """Parse a single logical proposition result into props and relations."""
        props = {}
        relations = []
        mode = None

        lines = result.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if 'Logical Propositions:' in line:
                mode = 'propositions'
                continue
            elif 'Logical Relations:' in line:
                mode = 'relations'
                continue

            if mode == 'propositions' and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and value:
                        props[key] = value
            elif mode == 'relations':
                if line and not line.lower().startswith('none'):
                    relations.append(line)

        return props, relations

    def _aggregate_logical_propositions(self, results: list) -> str:
        """Merge logical propositions from individual sentences."""
        all_props = {}
        all_relations = []
        prop_counter = ord('A')

        for result in results:
            props, relations = self._parse_logical_result(result)
            # Remap proposition keys to avoid conflicts
            key_mapping = {}
            for old_key, value in props.items():
                # Check if this proposition already exists
                existing_key = None
                for k, v in all_props.items():
                    if v.lower() == value.lower():
                        existing_key = k
                        break

                if existing_key:
                    key_mapping[old_key] = existing_key
                else:
                    new_key = chr(prop_counter)
                    prop_counter += 1
                    all_props[new_key] = value
                    key_mapping[old_key] = new_key

            # Update relations with remapped keys
            for rel in relations:
                updated_rel = rel
                for old_key, new_key in key_mapping.items():
                    updated_rel = re.sub(
                        r'\b' + re.escape(old_key) + r'\b',
                        lambda _: new_key,
                        updated_rel
                    )
                if updated_rel not in all_relations:
                    all_relations.append(updated_rel)

        # Format output
        output = "Logical Propositions:\n"
        for key in sorted(all_props.keys()):
            output += f"{key}: {all_props[key]}\n"
        output += "\nLogical Relations:\n"
        if all_relations:
            for rel in all_relations:
                output += f"{rel}\n"
        else:
            output += "None logical relations\n"

        return output

    def _extract_predicates_from_logical_form(self, logical_form: str) -> dict:
        """
        Extract predicates from logical form and determine their type signatures.

        Parses predicates like:
        - Grass(x) → bool => bool
        - Agent(e, x) → bool => bool => bool
        - Playing(e) → bool => bool
        - InFrontOf(x, z) → bool => bool => bool

        Args:
            logical_form: String containing logical form with predicates

        Returns:
            dict mapping predicate names to their Isabelle type signatures
        """
        predicates = {}

        # Pattern to match predicates: PredicateName(arg1, arg2, ...)
        predicate_pattern = r'([A-Z][a-zA-Z0-9]*)\s*\(([^)]+)\)'

        matches = re.findall(predicate_pattern, logical_form)

        for pred_name, args_str in matches:
            # Skip logical operators that might look like predicates
            if pred_name in ('And', 'Or', 'Not', 'Implies', 'Equivalent'):
                continue

            # Parse arguments
            args = [arg.strip() for arg in args_str.split(',')]
            args = [arg for arg in args if arg]  # Remove empty strings

            if not args:
                continue

            # Determine type signature based on arguments
            # All entity arguments use polymorphic type 'a
            # Return type is always bool (predicates return truth values)
            type_parts = ["'a"] * len(args)

            # Build Isabelle type signature: 'a => 'a => ... => bool
            type_signature = ' => '.join(type_parts + ['bool'])

            # Store the predicate (use the most general type if seen multiple times)
            if pred_name not in predicates:
                predicates[pred_name] = type_signature
            else:
                # If we've seen this predicate with different arities, use the longer one
                existing_arity = predicates[pred_name].count('=>')
                new_arity = type_signature.count('=>')
                if new_arity > existing_arity:
                    predicates[pred_name] = type_signature

        return predicates

    def _extract_typed_predicates(self, text: str) -> tuple:
        """
        Extract typed predicates and definitions from Predicates: lines in the text.

        Parses lines like:
        Predicates: Woman :: "agent => bool" -- "a classifier for women", Playing :: "agent => 'a => bool" -- "Playing(x, y) means agent x is playing y"

        Args:
            text: String containing one or more Predicates: lines

        Returns:
            tuple of (predicates, definitions) where:
              predicates: dict mapping predicate names to type signatures
              definitions: dict mapping predicate names to definition strings
        """
        predicates = {}
        definitions = {}

        # Find all Predicates: lines
        pred_line_matches = re.findall(r'Predicates:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)

        for pred_content in pred_line_matches:
            # Parse: Name :: "type" optionally followed by -- "definition"
            pred_matches = re.findall(r'(\w+)\s*::\s*"([^"]+)"(?:\s*--\s*"([^"]*)")?', pred_content)
            for match in pred_matches:
                name = match[0]
                type_sig = match[1]
                defn = match[2] if len(match) > 2 else ''

                if name not in predicates:
                    predicates[name] = type_sig
                    if defn:
                        definitions[name] = defn
                else:
                    # Keep higher arity version
                    existing_arity = predicates[name].count('=>')
                    new_arity = type_sig.count('=>')
                    if new_arity > existing_arity:
                        predicates[name] = type_sig
                    # Update definition if we got a new one
                    if defn and name not in definitions:
                        definitions[name] = defn

        return predicates, definitions

    def _extract_predicates_from_davidsonian(self, davidsonian_form: str) -> dict:
        """
        Extract all predicates from the entire Davidsonian form output.

        Prefers typed predicates from Predicates: lines over regex-extracted
        ones (which default to 'a for all arguments).

        Args:
            davidsonian_form: Full Davidsonian form string with logical forms

        Returns:
            dict mapping predicate names to their Isabelle type signatures
        """
        all_predicates = {}

        # First extract from logical forms via regex (default 'a types)
        logical_form_pattern = r'Logical form:\s*(.+?)(?:\n|$)'
        logical_forms = re.findall(logical_form_pattern, davidsonian_form, re.IGNORECASE)

        for lf in logical_forms:
            if lf.lower().strip() == 'none':
                continue
            preds = self._extract_predicates_from_logical_form(lf)
            for name, type_sig in preds.items():
                if name not in all_predicates:
                    all_predicates[name] = type_sig
                else:
                    existing_arity = all_predicates[name].count('=>')
                    new_arity = type_sig.count('=>')
                    if new_arity > existing_arity:
                        all_predicates[name] = type_sig

        # Overlay with typed predicates from Predicates: lines (these take priority)
        typed_preds, _ = self._extract_typed_predicates(davidsonian_form)
        for name, type_sig in typed_preds.items():
            if name in all_predicates:
                # Only override if arity matches or is higher
                existing_arity = all_predicates[name].count('=>')
                new_arity = type_sig.count('=>')
                if new_arity >= existing_arity:
                    all_predicates[name] = type_sig
            else:
                all_predicates[name] = type_sig

        return all_predicates

    def _convert_logical_form_to_isabelle(self, logical_form: str) -> str:
        """
        Convert a logical form string to Isabelle syntax.

        Converts:
        - Predicate(x, y) → Predicate x y (removes parentheses and commas)
        - \\<forall>, \\<exists>, \\<and>, \\<or>, \\<not> stay as is
        - --> stays as is

        Args:
            logical_form: Logical form string like "\\<forall>x. Violin(x) --> Instrument(x)"

        Returns:
            Isabelle-compatible axiom string
        """
        result = logical_form.strip()

        # Convert Predicate(arg1, arg2, ...) to Predicate arg1 arg2 ...
        # Pattern matches PredicateName followed by parentheses with arguments
        def replace_predicate(match):
            pred_name = match.group(1)
            args_str = match.group(2)
            args = [arg.strip() for arg in args_str.split(',')]
            return pred_name + ' ' + ' '.join(args)

        result = re.sub(r'([A-Z][a-zA-Z0-9]*)\s*\(([^)]+)\)', replace_predicate, result)

        return result

    def _add_missing_quantifiers(self, isabelle_formula: str) -> str:
        """Add explicit universal quantifiers for any free variables in the formula.

        In Isabelle axiomatizations, free variables become implicitly schematic
        (universally quantified). This works for standalone formulas, but breaks
        when the formula must unify with a quantified term inside a scheme
        predicate — e.g., a subgoal like PropositionInDomain (\\<forall>x y. P x y) d
        will NOT unify with a premise that has PropositionInDomain (P ?x ?y) d
        because \\<forall>x y. P x y is a different HOL term from P ?x ?y.
        """
        # Find all variables bound by quantifiers (\\<forall>x y. or \\<exists>x.)
        bound_vars = set()
        for match in re.finditer(r'\\<(?:forall|exists)>\s*([\w\s]+?)\.', isabelle_formula):
            vars_str = match.group(1)
            bound_vars.update(v.strip() for v in vars_str.split() if v.strip())

        # Find variable candidates: lowercase tokens of 1-2 characters
        # (typical variable names: x, y, z, d, s, a, e, e1, x1, etc.)
        all_vars = set()
        for match in re.finditer(r'\b([a-z][a-z0-9]?)\b', isabelle_formula):
            all_vars.add(match.group(1))

        free_vars = all_vars - bound_vars

        if free_vars:
            sorted_vars = ' '.join(sorted(free_vars))
            return f'\\<forall>{sorted_vars}. {isabelle_formula}'

        return isabelle_formula

    # ---- Isar proof generation methods ----

    def _parse_scheme_premises(self, scheme: dict) -> dict:
        """Parse scheme axiom to extract premises, conclusion, metavars, rule name.

        Handles formats:
          '[| P1; P2; P3 |] ==> C'
          'P ==> C'
          With \\<not>, !!y quantifiers, etc.

        Returns: {
            'rule_name': str,
            'premises': ['InPositionToKnow s d', ...],
            'conclusion': 'a',
            'metavars': {'s', 'd', 'a'}
        }
        """
        axiom_text = scheme.get('isabelle_axiom', '')
        rule_name = scheme.get('rule_name', '')

        # Extract the formula from the axiomatization block
        formula_match = re.search(
            r'axiomatization\s+where\s*\n?\s*\w+\s*:\s*"(.+)"',
            axiom_text, re.DOTALL)
        if not formula_match:
            logger.error(f"Could not parse scheme axiom: {axiom_text}")
            return {'rule_name': rule_name, 'premises': [],
                    'conclusion': '', 'metavars': set()}

        formula = formula_match.group(1).strip()

        # Split on the LAST ==> to get premises and conclusion.
        # Some schemes have ==> inside [| |] brackets (e.g., verbal_classification)
        # so we must split on the outermost one only.
        parts = formula.rsplit('==>', 1)
        if len(parts) != 2:
            logger.error(f"No ==> found in scheme formula: {formula}")
            return {'rule_name': rule_name, 'premises': [formula],
                    'conclusion': '', 'metavars': set()}

        lhs = parts[0].strip()
        conclusion = parts[1].strip()

        # Parse premises: strip [| |] brackets if present, split on ;
        if '[|' in lhs and '|]' in lhs:
            inner = lhs[lhs.index('[|') + 2:lhs.rindex('|]')].strip()
            premises = [p.strip() for p in inner.split(';') if p.strip()]
        else:
            premises = [lhs]

        # Extract metavars: lowercase tokens of 1-2 chars that appear
        # in premises/conclusion but are NOT Isabelle keywords or
        # bound variables from metalevel quantifiers (!! or \<forall>)
        keywords = {'in', 'of', 'if', 'by', 'do', 'or', 'no', 'at', 'to',
                     'is', 'as', 'so', 'be', 'on', 'an'}
        all_text = ' '.join(premises) + ' ' + conclusion

        # Find bound variables from !! quantifiers (e.g., "!!y." binds y)
        bound_vars = set()
        for bm in re.finditer(r'!!\s*([\w\s]+?)\.', all_text):
            for v in bm.group(1).split():
                bound_vars.add(v)

        metavar_candidates = set()
        for match in re.finditer(r'\b([a-z][a-z0-9]?)\b', all_text):
            token = match.group(1)
            if token not in keywords and token not in bound_vars:
                metavar_candidates.add(token)

        logger.debug(f"Parsed scheme '{rule_name}': "
                     f"premises={premises}, conclusion={conclusion}, "
                     f"metavars={metavar_candidates}")

        return {
            'rule_name': rule_name,
            'premises': premises,
            'conclusion': conclusion,
            'metavars': metavar_candidates
        }

    def _generate_obtain_step(self, assumption_formula: str) -> dict:
        """Generate Isar obtain step from an existential assumption formula.

        For: \\<exists> x y. MedicalProfessional x \\<and> TerminallyIll y
        Returns: {
            'step': 'obtain x y where asm_facts: "MedicalProfessional x" "TerminallyIll y"
                     using asm by blast',
            'fact_name': 'asm_facts',
            'variables': ['x', 'y']
        }

        For non-existential: returns None.
        """
        if not assumption_formula:
            return None

        # Check for outermost existential quantifier
        exist_match = re.match(
            r'\\<exists>\s*([\w\s]+?)\.\s*(.+)', assumption_formula, re.DOTALL)
        if not exist_match:
            return None

        vars_str = exist_match.group(1).strip()
        body = exist_match.group(2).strip()
        variables = vars_str.split()

        # Split body on \<and> at top level into conjuncts
        # Simple split — doesn't handle nested parens with \<and> inside
        conjuncts = re.split(r'\\<and>', body)
        conjuncts = [c.strip() for c in conjuncts if c.strip()]

        if not conjuncts:
            conjuncts = [body]

        # Build the obtain step
        quoted_facts = ' '.join(f'"{c}"' for c in conjuncts)
        fact_name = 'asm_facts'
        step = (f'obtain {vars_str} where {fact_name}: {quoted_facts} '
                f'using asm by blast')

        logger.debug(f"Generated obtain step: {step}")

        return {
            'step': step,
            'fact_name': fact_name,
            'variables': variables
        }

    def _infer_metavar_types(self, premises: list, conclusion: str,
                             scheme_consts: dict) -> dict:
        """Infer the Isabelle type of each metavar from how predicates use them.

        For each predicate P in the scheme premises, look up P's type signature
        in scheme_consts and assign types to its positional arguments.
        E.g. SimilarTo :: "domain => domain => bool" with SimilarTo c1 c2
        → c1: domain, c2: domain.
        """
        metavar_types = {}
        all_text = premises + [conclusion]
        for expr in all_text:
            # Match predicate application: PredName arg1 arg2 ...
            m = re.match(r'(\w+)\s+(.*)', expr.strip())
            if not m:
                continue
            pred_name = m.group(1)
            args_str = m.group(2).strip()
            if pred_name not in scheme_consts:
                continue
            type_sig = scheme_consts[pred_name]
            # Parse type signature: extract argument types before "=> bool"
            arg_types = re.findall(r'"?(\w+)"?\s*=>', type_sig)
            # Split args (simple single-token args only)
            args = args_str.split()
            for arg, typ in zip(args, arg_types):
                # Only annotate single lowercase metavar-like tokens
                if re.match(r'^[a-z][a-z0-9]?$', arg):
                    metavar_types[arg] = typ
        return metavar_types

    def _get_metavar_bindings(self, parsed_scheme: dict, hypothesis_lf: str,
                              premise_lf: str, obtain_info: dict,
                              generated_premise_formulas: list,
                              error_context: str = '') -> dict:
        """Call LLM to get metavariable bindings for scheme instantiation.

        Returns dict like: {'s': 'x', 'd': 'Medicine', 'a': '...'}
        """
        # Format scheme premises
        premises_text = '\n'.join(
            f'{i+1}. {p}' for i, p in enumerate(parsed_scheme['premises']))

        # Format metavars with type annotations inferred from scheme consts
        scheme_consts = {}
        if hasattr(self, 'current_scheme') and self.current_scheme:
            raw = self.current_scheme.get('consts', {})
            for name, val in raw.items():
                scheme_consts[name] = val.get('type', val) if isinstance(val, dict) else val
        metavar_types = self._infer_metavar_types(
            parsed_scheme['premises'], parsed_scheme['conclusion'], scheme_consts)

        # Auto-bind conclusion metavar to shows clause (Fix 1)
        # The conclusion metavar (e.g. 'a' in "Scheme conclusion: a") must equal the shows
        # clause so that scheme_rule[OF step_1 ... step_n] produces exactly ?thesis.
        pre_bindings = {}
        remaining_metavars = set(parsed_scheme['metavars'])
        conclusion_mv = parsed_scheme['conclusion'].strip()
        if (conclusion_mv and ' ' not in conclusion_mv
                and conclusion_mv in remaining_metavars
                and hypothesis_lf):
            pre_bindings[conclusion_mv] = hypothesis_lf
            remaining_metavars.discard(conclusion_mv)
            logger.info(f"Auto-bound conclusion metavar '{conclusion_mv}' = '{hypothesis_lf[:60]}...'")

        metavars_text = ', '.join(
            f'{mv} (type: {metavar_types[mv]})' if mv in metavar_types else mv
            for mv in sorted(remaining_metavars)
        )

        # Format generated premise axioms
        gp_text = '\n'.join(
            f'{name}: "{formula}"' for name, formula in generated_premise_formulas)

        # Format obtain info
        obtain_text = ''
        if obtain_info:
            obtain_text = (f'Variables obtained from assumption: '
                          f'{", ".join(obtain_info["variables"])}')

        # Format scheme description
        scheme_description = ''
        if hasattr(self, 'current_scheme') and self.current_scheme:
            scheme_description = self.current_scheme.get('description', '')

        predicate_defs_text = self._build_predicate_defs_text()

        # If all metavars are pre-bound, skip LLM call
        llm_bindings = {}
        new_consts = {}
        if remaining_metavars:
            # NL argument text for grounding context
            nl_premise = getattr(self, '_nl_premise', None)
            nl_hypothesis = getattr(self, '_nl_hypothesis', None)
            if isinstance(nl_premise, list):
                nl_premise_text = ' '.join(str(p) for p in nl_premise if p)
            else:
                nl_premise_text = str(nl_premise) if nl_premise else '(none)'
            nl_hypothesis_text = str(nl_hypothesis) if nl_hypothesis else '(none)'

            # Call LLM for the remaining unbound metavars
            result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name=self.prompt_dict.get('instantiate scheme',
                                                 'instantiate_scheme_prompt.txt'),
                no_code_extract=True,
                rule_name=parsed_scheme['rule_name'],
                scheme_premises=premises_text,
                conclusion=parsed_scheme['conclusion'],
                metavars=metavars_text,
                scheme_description=scheme_description,
                predicate_definitions=predicate_defs_text,
                nl_premise=nl_premise_text,
                nl_hypothesis=nl_hypothesis_text,
                hypothesis=hypothesis_lf or '(none)',
                assumption=premise_lf or '(none)',
                obtain_info=obtain_text,
                generated_premises=gp_text or '(none)',
                error_context=error_context
            )

            logger.debug(f"LLM metavar bindings output:\n{result}")

            # Parse new_const declarations and key=value bindings
            for line in result.strip().split('\n'):
                line = line.strip()
                if line.startswith('new_const:'):
                    decl = line[len('new_const:'):].strip()
                    m = re.match(r'(\w+)\s*::\s*(.+)', decl)
                    if m:
                        const_name = m.group(1).strip()
                        const_type = m.group(2).strip().strip('"')
                        new_consts[const_name] = const_type
                        logger.info(f"New domain const declared: {const_name} :: {const_type}")
                elif '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        llm_bindings[key] = value

            # Inject new consts into the current theory code
            if new_consts and self.code:
                new_const_lines = '\n'.join(
                    f'  {name} :: "{typ}"' for name, typ in new_consts.items()
                )
                self.code = re.sub(
                    r'(consts\b.*?\n)((?:  \S.*\n)*)',
                    lambda m: m.group(0) + new_const_lines + '\n',
                    self.code, count=1, flags=re.DOTALL
                )
                logger.info(f"Injected new consts into theory: {list(new_consts.keys())}")

        # Merge: pre_bindings take priority (LLM must not override them)
        bindings = {**llm_bindings, **pre_bindings}
        logger.info(f"Metavar bindings: {bindings}")
        return bindings

    def _apply_metavar_bindings(self, premises: list, bindings: dict) -> list:
        """Substitute metavar bindings into scheme premises.

        Complex values (containing spaces/operators) are wrapped in parens.
        Replacement uses word boundaries to avoid partial matches.
        """
        instantiated = []
        for premise in premises:
            result = premise
            # Sort by length descending to avoid e.g. 'c1' matching 'c' first
            for metavar in sorted(bindings.keys(), key=len, reverse=True):
                value = bindings[metavar]
                # Wrap complex values in parentheses
                is_complex = (' ' in value or '\\<' in value or
                              '-->' in value or '(' in value)
                if is_complex:
                    replacement = f'({value})'
                else:
                    replacement = value
                # Replace as whole word
                result = re.sub(rf'\b{re.escape(metavar)}\b', lambda _, r=replacement: r, result)
            instantiated.append(result)

        logger.debug(f"Instantiated premises: {instantiated}")
        return instantiated

    @staticmethod
    def _parse_shows_quantifiers(shows_clause: str):
        """Parse a shows clause of the form \\<forall>x y. A x y --> B x y --> C x y.

        Returns (fixed_vars, antecedents, consequent):
            fixed_vars:  list of variable names bound by leading \\<forall>
            antecedents: list of antecedent strings (one per top-level -->),
                         empty if no top-level implication
            consequent:  the final consequent (no more top-level -->)
        """
        if not shows_clause:
            return [], [], shows_clause

        s = shows_clause.strip()

        # Strip leading \<forall> binders
        fixed_vars = []
        forall_pat = re.compile(r'^\\<forall>\s*([\w\s]+?)\s*\.\s*')
        while True:
            m = forall_pat.match(s)
            if not m:
                break
            fixed_vars.extend(v.strip() for v in m.group(1).split() if v.strip())
            s = s[m.end():]

        body = s.strip()
        if body.startswith('(') and body.endswith(')'):
            body = body[1:-1].strip()

        # If the body starts with \<exists>, the --> is inside the existential
        # scope — treat the whole body as the consequent so we use proof -
        # rather than intro impI (which would fail on an existential goal).
        if body.startswith(r'\<exists>'):
            return fixed_vars, [], body

        # Strip ALL top-level --> antecedents iteratively
        antecedents = []
        while True:
            depth = 0
            arrow_pos = None
            i = 0
            while i < len(body) - 2:
                c = body[i]
                if c in '([':
                    depth += 1
                elif c in ')]':
                    depth -= 1
                elif depth == 0 and body[i:i+3] == '-->':
                    arrow_pos = i
                    break
                i += 1
            if arrow_pos is None:
                break
            antecedents.append(body[:arrow_pos].strip())
            body = body[arrow_pos + 3:].strip()

        return fixed_vars, antecedents, body

    def _generate_isar_proof(self, parsed_scheme: dict,
                             instantiated_premises: list,
                             axiom_names: list, obtain_info: dict,
                             has_assumption: bool,
                             asm_labels: list = None,
                             shows_clause: str = None) -> str:
        """Generate complete Isar proof block.

        Each have step uses ALL generated premises + obtained facts.
        Uses 'sorry' as placeholder — critique loop replaces with real tactics.

        When shows_clause has leading \\<forall> quantifiers, generates:
          proof (intro allI [impI])
            fix x y
            assume asm_inner: "antecedent"
            have ...
            show "consequent" ...
          qed
        so that fixed variables are in scope for all have steps.
        """
        fixed_vars, antecedents, consequent = \
            self._parse_shows_quantifiers(shows_clause)

        needs_fix    = bool(fixed_vars)
        needs_assume = bool(antecedents)

        if needs_fix and needs_assume:
            proof_opener = 'proof (intro allI impI)'
        elif needs_fix:
            proof_opener = 'proof (intro allI)'
        elif needs_assume:
            proof_opener = 'proof (intro impI)'
        else:
            proof_opener = 'proof -'

        lines = [proof_opener]
        indent = '  '

        # Fix universally quantified variables into scope
        if needs_fix:
            lines.append(f'{indent}fix {" ".join(fixed_vars)}')

        # One assume per top-level antecedent
        inner_asm_labels = []
        for k, ant in enumerate(antecedents):
            label = 'asm_inner' if k == 0 else f'asm_inner_{k}'
            inner_asm_labels.append(label)
            lines.append(f'{indent}assume {label}: "{ant}"')

        # Obtain step (if existential assumption)
        if obtain_info:
            lines.append(f'{indent}{obtain_info["step"]}')

        # Build using clause: generated premises + assumption references
        using_parts = list(axiom_names)
        if obtain_info:
            using_parts.append(obtain_info['fact_name'])
            obtained_label = obtain_info.get('asm_label', 'asm')
            if asm_labels:
                for label in asm_labels:
                    if label != obtained_label:
                        using_parts.append(label)
        elif has_assumption:
            if asm_labels:
                using_parts.extend(asm_labels)
            else:
                using_parts.append('asm')
        using_parts.extend(inner_asm_labels)
        using_clause = ' '.join(using_parts) if using_parts else ''

        # Have steps — one per instantiated premise
        step_labels = []
        for i, premise in enumerate(instantiated_premises):
            label = f'step_{i + 1}'
            step_labels.append(label)
            if using_clause:
                lines.append(f'{indent}have {label}: "{premise}"')
                lines.append(f'{indent}  using {using_clause}')
                lines.append(f'{indent}  sorry')
            else:
                lines.append(f'{indent}have {label}: "{premise}"')
                lines.append(f'{indent}  sorry')

        # Show step — apply scheme rule with [OF step_1 step_2 ...]
        rule_name = parsed_scheme['rule_name']
        of_clause = ' '.join(step_labels)
        # Use explicit consequent when we decomposed a \<forall>/-->
        show_target = f'"{consequent}"' if (needs_fix or needs_assume) else '?thesis'
        lines.append(f'{indent}show {show_target}')
        lines.append(f'{indent}  using {rule_name}[OF {of_clause}]')
        lines.append(f'{indent}  sorry')

        lines.append('qed')

        proof_text = '\n'.join(lines)
        logger.debug(f"Generated Isar proof:\n{proof_text}")
        return proof_text

    def regenerate_isar_proof(self, isabelle_code: str,
                              error_messages: list,
                              unsolved_goals: list = None) -> Optional[str]:
        """Regenerate Isar proof with fresh metavar bindings.

        Called when type errors or stagnation indicate bad bindings.
        Passes the error messages and unsolved goals to the LLM.
        Returns updated code or None if regeneration isn't possible.
        """
        scheme = getattr(self, 'current_scheme', None)
        if not scheme:
            logger.debug("No current_scheme — cannot regenerate proof")
            return None

        parsed = self._parse_scheme_premises(scheme)
        if not parsed:
            logger.debug("Failed to parse scheme premises")
            return None

        # Re-extract context from the existing code
        axiom_names = [a['name']
                       for a in self.extract_generated_axioms(isabelle_code)]
        gp_formulas = [(a['name'], a['formula'])
                       for a in self.extract_generated_axioms(isabelle_code)]

        # Extract hypothesis and all assumptions from the code
        hyp_match = re.search(r'shows\s+"([^"]+)"', isabelle_code)
        hypothesis_lf = hyp_match.group(1) if hyp_match else ''

        # Extract all assumes/and clauses
        premise_lfs = []
        asm_labels = []
        asm_pattern = re.compile(
            r'(?:assumes|and)\s+(\w+):\s+"([^"]+)"')
        for m in asm_pattern.finditer(isabelle_code):
            asm_labels.append(m.group(1))
            premise_lfs.append(m.group(2))
        all_assumptions_text = '\n'.join(
            f'{label}: "{plf}"'
            for label, plf in zip(asm_labels, premise_lfs)
        ) if premise_lfs else None

        # Generate obtain step from first existential assumption
        obtain_info = None
        for idx, plf in enumerate(premise_lfs):
            info = self._generate_obtain_step(plf)
            if info:
                info['step'] = info['step'].replace(
                    'using asm ', f'using {asm_labels[idx]} ')
                info['asm_label'] = asm_labels[idx]
                obtain_info = info
                break

        # Format error messages as context for the LLM.
        # Both type unification failures and identity bindings (where a metavar
        # was bound to itself, e.g. d -> d, leaving it as a fixed variable in
        # the goal rather than a concrete term) are binding errors.
        error_context = ''
        type_errors = [m for m in (error_messages or [])
                       if 'Type unification failed' in m]
        metavar_names = parsed.get('metavars', set())
        identity_bound_goals = [
            g for g in (unsolved_goals or [])
            if any(re.search(rf'\b{re.escape(mv)}\b', g)
                   for mv in metavar_names)
        ]
        prev_bindings = getattr(self, '_last_metavar_bindings', {})

        if type_errors or identity_bound_goals:
            error_context = (
                "IMPORTANT: The previous bindings produced type/binding errors "
                "in Isabelle. Do NOT repeat the same bindings — use concrete "
                "terms for metavars, NOT raw metavar names or predicate "
                "applications like DrJackKevorkian x:\n"
                + '\n'.join(type_errors)
            )
            if prev_bindings:
                prev_text = '\n'.join(
                    f'  {k} = {v}' for k, v in prev_bindings.items())
                error_context += (
                    "\nPrevious (failed) bindings — do not reuse:\n" + prev_text
                )
            if identity_bound_goals:
                error_context += (
                    "\nThe following have-step goals still contain uninstantiated "
                    "metavar names — the metavar was bound to itself rather than "
                    "a concrete term. Choose bindings that replace these with "
                    "actual entities from the premises:\n"
                    + '\n'.join(f'  {g}' for g in identity_bound_goals)
                )
        if unsolved_goals:
            goals_text = '\n'.join(f'  {g}' for g in unsolved_goals)
            error_context += (
                "\nIMPORTANT: The previous bindings produced these unsolved "
                "have steps — the goals below could not be proved even after "
                "multiple refinement attempts. Choose different bindings that "
                "make these goals provable from the assumption:\n"
                + goals_text
            )

        # Get fresh metavar bindings with error context
        bindings = self._get_metavar_bindings(
            parsed, hypothesis_lf, all_assumptions_text, obtain_info, gp_formulas,
            error_context=error_context
        )

        if not bindings:
            logger.warning("No metavar bindings returned on regeneration")
            return None

        # Apply bindings and generate new proof
        instantiated = self._apply_metavar_bindings(
            parsed['premises'], bindings)
        proof_text = self._generate_isar_proof(
            parsed, instantiated, axiom_names, obtain_info,
            has_assumption=bool(premise_lfs),
            asm_labels=asm_labels,
            shows_clause=hypothesis_lf
        )

        # Replace proof block in the code (handles both proof - and proof (intro ...))
        updated = re.sub(
            r'proof\b.*?qed',
            lambda _, t=proof_text: t,
            isabelle_code,
            flags=re.DOTALL
        )

        updated = self._fix_unicode_symbols(updated)
        logger.info(f"Regenerated Isar proof with bindings: {bindings}")
        return updated

    def _extract_logical_forms_by_section(self, davidsonian_form: str) -> dict:
        """
        Extract logical forms organized by section (hypothesis, generated premise, existing premise).

        Args:
            davidsonian_form: Full Davidsonian form string

        Returns:
            dict with keys 'hypothesis', 'generated_premises', 'existing_premises'
            each containing a list of (order, logical_form) tuples
        """
        result = {
            'hypothesis': [],
            'generated_premises': [],
            'existing_premises': []
        }

        lines = davidsonian_form.split('\n')
        current_section = None
        current_order = 0

        for line in lines:
            line_lower = line.lower().strip()

            # Detect section headers
            if line_lower.startswith('hypothesis'):
                current_section = 'hypothesis'
                current_order = 0
            elif line_lower.startswith('generated premise'):
                current_section = 'generated_premises'
                current_order = 0
            elif line_lower.startswith('existing premise') or line_lower.startswith('premise sentence'):
                current_section = 'existing_premises'
                current_order = 0

            # Check for numbered lines (e.g., "1. ...")
            num_match = re.match(r'^(\d+)\.', line.strip())
            if num_match:
                current_order = int(num_match.group(1))

            # Extract logical form (handles both "Logical form: ..." and "1. Logical form: ...")
            lf_match = re.match(r'^(?:\d+\.\s*)?logical\s+form:\s*(.*)', line.strip(), re.IGNORECASE)
            if lf_match:
                lf = lf_match.group(1).strip()
                if lf.lower() != 'none' and current_section:
                    result[current_section].append((current_order, lf))

        return result

    def _generate_axioms_from_logical_forms(self, davidsonian_form: str) -> list:
        """
        Generate Isabelle axiom blocks directly from extracted logical forms.

        Args:
            davidsonian_form: Full Davidsonian form string

        Returns:
            List of axiom block strings
        """
        logical_forms = self._extract_logical_forms_by_section(davidsonian_form)
        axiom_blocks = []

        # Generate axioms for generated premises
        for order, lf in logical_forms['generated_premises']:
            isabelle_lf = self._convert_logical_form_to_isabelle(lf)
            isabelle_lf = self._add_missing_quantifiers(isabelle_lf)
            axiom_block = f'(* Generated Premise {order} *)\n'
            axiom_block += f'axiomatization where\n'
            axiom_block += f'  generated_premise_{order}: "{isabelle_lf}"'
            axiom_blocks.append(axiom_block)

        return axiom_blocks

    def extract_generated_axioms(self, isabelle_code: str) -> list[dict]:
        """Extract generated premise axioms from Isabelle code.

        Returns list of dicts with 'name' and 'formula' keys.
        """
        axioms = []
        pattern = re.compile(
            r'\(\*\s*Generated Premise\s+(\d+)\s*\*\)\s*\n'
            r'\s*axiomatization where\s*\n'
            r'\s*(generated_premise_\d+):\s*"([^"]*)"',
            re.DOTALL
        )
        for match in pattern.finditer(isabelle_code):
            axioms.append({
                'name': match.group(2),
                'formula': match.group(3),
            })
        return axioms

    def extract_bridge_axioms(self, isabelle_code: str) -> list[dict]:
        """Extract bridge axioms from Isabelle code.

        Returns list of dicts with 'name' and 'formula' keys.
        Bridge axioms are marked with (* Bridge axioms for scheme instantiation *).
        """
        axioms = []
        bridge_match = re.search(
            r'\(\*\s*Bridge axioms.*?\*\)\s*\n(.*?)(?=\ntheorem|\n\(\*)',
            isabelle_code, re.DOTALL)
        if not bridge_match:
            return axioms
        bridge_block = bridge_match.group(1)
        for m in re.finditer(r'(\w+):\s*"([^"]*)"', bridge_block):
            axioms.append({'name': m.group(1), 'formula': m.group(2)})
        return axioms

    def replace_generated_axioms(self, isabelle_code: str,
                                  new_axiom_blocks: str) -> str:
        """Remove all generated_premise_* axiom blocks and insert new ones.

        Also updates apply (insert generated_premise_*) lines in the proof.
        Strips any consts blocks from new_axiom_blocks to prevent duplicate
        constant declarations (add_consts_if_needed handles new consts separately).
        """
        # Remove existing generated premise blocks
        cleaned = re.sub(
            r'\(\*\s*Generated Premise\s+\d+\s*\*\)\s*\n'
            r'\s*axiomatization where\s*\n'
            r'\s*generated_premise_\d+:\s*"[^"]*"\s*\n?',
            '',
            isabelle_code
        )

        # Strip consts blocks from new_axiom_blocks to avoid duplicating
        # declarations that already exist in the theory
        cleaned_axioms = re.sub(
            r'consts\s*\n(?:\s+\w+\s*::.*\n)*',
            '',
            new_axiom_blocks
        ).strip()

        # Insert new axiom blocks before the theorem
        theorem_match = re.search(r'^theorem\s', cleaned, re.MULTILINE)
        if theorem_match:
            insert_pos = theorem_match.start()
            cleaned = (cleaned[:insert_pos].rstrip('\n') + '\n\n'
                       + cleaned_axioms + '\n\n'
                       + cleaned[insert_pos:])

        # Update apply (insert ...) lines to match new axiom names
        new_names = re.findall(r'(generated_premise_\d+):', new_axiom_blocks)
        # Remove old insert lines (use [ \t]* not \s* to avoid consuming
        # newlines, which would collapse apply (rule ...) and oops onto one line)
        cleaned = re.sub(
            r'[ \t]*apply \(insert generated_premise_\d+\)[ \t]*\n?',
            '',
            cleaned
        )
        # Build new insert lines
        if new_names:
            insert_lines = '\n'.join(
                f'  apply (insert {name})' for name in new_names
            )
            # Insert before sledgehammer (or before oops if no sledgehammer)
            sledge_match = re.search(r'^(\s*)sledgehammer', cleaned, re.MULTILINE)
            if sledge_match:
                cleaned = (cleaned[:sledge_match.start()]
                           + insert_lines + '\n'
                           + cleaned[sledge_match.start():])
            else:
                oops_match = re.search(r'^(\s*)oops', cleaned, re.MULTILINE)
                if oops_match:
                    cleaned = (cleaned[:oops_match.start()]
                               + insert_lines + '\n'
                               + cleaned[oops_match.start():])

        return cleaned

    def add_consts_if_needed(self, isabelle_code: str,
                              new_axiom_text: str) -> str:
        """Add any new predicate consts that appear in axiom formulas
        but are missing from the existing consts section."""
        # Extract existing consts
        existing_consts = set()
        for match in re.finditer(r'^\s+(\w+)\s*::', isabelle_code, re.MULTILINE):
            existing_consts.add(match.group(1))

        # Extract predicates used in new axiom formulas (capitalised words
        # that appear at predicate positions — before arguments)
        used_preds = set()
        for match in re.finditer(r'(generated_premise_\d+):\s*"([^"]*)"',
                                  new_axiom_text):
            formula = match.group(2)
            # Find capitalised identifiers that look like predicates
            for pred_match in re.finditer(r'\b([A-Z]\w*)\b', formula):
                pred = pred_match.group(1)
                # Skip Isabelle keywords and quantifier variables
                if pred not in {'True', 'False', 'Main', 'HOL', 'UNIV',
                                'Set', 'None', 'Some', 'Not', 'And', 'Or'}:
                    used_preds.add(pred)

        missing = used_preds - existing_consts
        if not missing:
            return isabelle_code

        # Build new const declarations (default to bool => bool)
        new_const_lines = '\n'.join(
            f'  {pred} :: "bool => bool"' for pred in sorted(missing)
        )

        # Insert after existing consts block
        consts_match = re.search(r'^consts\n((?:\s+\w+\s*::.*\n)*)',
                                  isabelle_code, re.MULTILINE)
        if consts_match:
            insert_pos = consts_match.end()
            isabelle_code = (isabelle_code[:insert_pos]
                             + new_const_lines + '\n'
                             + isabelle_code[insert_pos:])

        return isabelle_code

    def _parse_axiom_code(self, axiom_code: str) -> tuple:
        """Parse Isabelle axiom code to extract consts, axiom block, and definitions.

        Returns (consts, axiom_block, definitions) where:
          consts      -- dict of name -> type
          axiom_block -- the axiomatization block as a string
          definitions -- dict of name -> definition string (from -- "..." annotations)
        """
        consts = {}
        definitions = {}
        axiom_block = ""

        lines = axiom_code.split('\n')
        in_consts = False
        axiom_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip boilerplate
            if stripped.startswith(('begin', 'end', 'typedecl', 'imports', 'theory')):
                continue

            if stripped == 'consts':
                in_consts = True
                continue

            if in_consts:
                if '::' in stripped:
                    # Parse const declaration with optional -- "definition"
                    match = re.match(r'(\w+)\s*::\s*"([^"]+)"(?:\s*--\s*"([^"]*)")?', stripped)
                    if match:
                        const_name = match.group(1)
                        consts[const_name] = match.group(2)
                        if match.group(3):
                            definitions[const_name] = match.group(3)
                elif stripped.startswith('(*') or stripped.startswith('axiomatization'):
                    in_consts = False

            if stripped.startswith('(*') or stripped.startswith('axiomatization'):
                in_consts = False
                axiom_lines.append(line)
            elif not in_consts and stripped and not stripped.startswith(('begin', 'end', 'typedecl', 'imports', 'theory', 'consts')):
                if '::' not in stripped:  # Not a const declaration
                    axiom_lines.append(line)

        axiom_block = '\n'.join(axiom_lines).strip()
        return consts, axiom_block, definitions

    def _merge_axiom_code(self, all_consts: dict, axiom_blocks: list,
                          scheme_axiom: str = None, scheme_consts: dict = None,
                          scheme_name: str = None) -> str:
        """Merge multiple axiom codes into a single Isabelle code block."""
        # Merge scheme consts with extracted consts
        if scheme_consts:
            all_consts = {**all_consts, **scheme_consts}

        # Build consts section
        consts_lines = []
        for name in sorted(all_consts.keys()):
            consts_lines.append(f'  {name} :: "{all_consts[name]}"')

        consts_section = "consts\n" + '\n'.join(consts_lines)

        # Combine axiom blocks
        axioms_section = '\n\n'.join(block for block in axiom_blocks if block.strip())

        # Add argument scheme axiom if provided
        if scheme_axiom:
            axioms_section += f'\n\n(* Argument Scheme: {scheme_name or "unknown"} *)\n{scheme_axiom.strip()}'

        # Type declarations for the polymorphic type system
        typedecls = "typedecl agent\ntypedecl domain"

        result = (
            "imports Main\n\n" +
            "begin\n\n" +
            typedecls + "\n\n" +
            consts_section + "\n\n" +
            axioms_section +
            "\n\ntheorem hypothesis:\n assumes asm: \n shows \nproof -\n  \n  \nqed\n\nend"
        )

        logger.debug(f"_merge_axiom_code: consts={len(all_consts)}, axiom_blocks={len(axiom_blocks)}, result_len={len(result)}")
        return result

    def _extract_generated_premises_from_davidsonian(self, davidsonian_form: str) -> list:
        """Extract individual generated premises from Davidsonian form."""
        lower_case = davidsonian_form.lower()
        start_idx = lower_case.find("generated premise")
        if start_idx == -1:
            return []

        # Find end of generated premises section
        end_idx = lower_case.find("premise sentence", start_idx)
        if end_idx == -1:
            end_idx = lower_case.find("existing premise", start_idx)
        if end_idx == -1:
            end_idx = len(davidsonian_form)

        gen_section = davidsonian_form[start_idx:end_idx]

        # Split by numbered items
        premises = []
        # Pattern to match numbered premises with their logical forms
        pattern = r'(\d+\.\s*[^\n]+(?:\n(?!(?:\d+\.|Generated|Premise|Existing|Hypothesis))[^\n]*)*)'
        matches = re.findall(pattern, gen_section, re.MULTILINE)

        for match in matches:
            if 'logical form' in match.lower():
                premises.append(match.strip())

        return premises

    # ==================== Single-Sentence Processing Methods ====================

    def _flatten_parse_result(self, result: str) -> str:
        """Strip deeply nested parse lines, keeping only top-level constituents."""
        lines = result.split('\n')
        filtered = []
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Keep: top-level (indent 0) and first level of sub-items (indent 1-4)
            if indent <= 4:
                filtered.append(line)
        return '\n'.join(filtered)

    def _get_parsing_single(self, sentence: str, sentence_type: str, order: int) -> str:
        """Parse a single sentence. Results are cached by sentence content."""
        cache_key = sentence.strip()
        if cache_key in self._parsing_cache:
            logger.debug(f"Using cached parsing for {sentence_type} {order}")
            return self._parsing_cache[cache_key]

        formatted = f"{sentence_type}:\n{order}. {sentence.strip()}\n"

        try:
            result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name=self.prompt_dict['get sentence parse'],
                input_sentence=formatted
            )

            if result is None or not result.strip():
                logger.warning(f"Empty result for sentence: {sentence[:50]}...")
                return ""

            # Clean up the result
            result = re.sub(r'^.*?answer:\s*', '', result,
                           flags=re.DOTALL | re.IGNORECASE)
            result = result.strip()
            result = self._flatten_parse_result(result)

            if result:
                self._parsing_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error parsing sentence '{sentence[:50]}...': {e}")
            return ""

    def _get_davidsonian_single(self, parsed_sentence: str, sentence_type: str,
                                existing_predicates: str = '') -> str:
        """Convert a single parsed sentence to Davidsonian form.

        Results for fixed sentence types (hypothesis, existing premise) are
        cached so they are not re-processed across refinement iterations.
        """
        is_fixed = sentence_type != "Generated Premise"
        cache_key = parsed_sentence.strip()

        if is_fixed and cache_key in self._davidsonian_cache:
            logger.debug(f"Using cached Davidsonian form for {sentence_type}")
            return self._davidsonian_cache[cache_key]

        try:
            result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name=self.prompt_dict['get davidsonian'],
                input_sentence=parsed_sentence,
                existing_predicates=existing_predicates
            )
            result = result.strip() if result else ""

            if is_fixed and result:
                self._davidsonian_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Error getting Davidsonian form: {e}")
            return ""

    def _get_logical_proposition_single(self, sentence: str, order: int) -> str:
        """Extract logical propositions from a single generated premise."""
        formatted = f"Generated Premise {order}: {sentence.strip()}\n"

        try:
            result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name=self.prompt_dict['get logical proposition'],
                generated_premises=formatted
            )
            return result.strip() if result else ""
        except Exception as e:
            logger.error(f"Error getting logical proposition: {e}")
            return ""

    def _get_axiom_single(self, davidsonian_premise: str, order: int) -> str:
        """Generate Isabelle axiom for a single generated premise."""
        formatted = f"Generated Premise {order}:\n{davidsonian_premise}"

        try:
            result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name=self.prompt_dict['get isabelle axiom'],
                explanatory_sentences=formatted
            )
            return result.strip() if result else ""
        except Exception as e:
            logger.error(f"Error generating axiom: {e}")
            return ""

    # ==================== Refactored Main Methods ====================

    def _get_parsing(self, premise: list, generated_premises: list,
                     hypothesis: str) -> tuple:
        """
        Parse sentences one by one and return structured results.

        Returns:
            tuple: (aggregated_string, results_list) where results_list is
                   [(sentence_type, order, parsed_result), ...]
        """
        results = []

        # Process fixed sentences first so their predicates condition
        # generated premises (processed last in davidsonian form step).

        # 1. Parse hypothesis sentences
        hypothesis_sentences = [hypothesis]
        logger.info(f"Parsing {len(hypothesis_sentences)} hypothesis sentence(s)")
        for i, sent in enumerate(hypothesis_sentences, 1):
            logger.debug(f"Hypothesis {i}: {sent[:50]}...")
            result = self._get_parsing_single(sent, "Hypothesis Sentence", i)
            if result:
                results.append(("Hypothesis Sentence", i, result))
                log_llm_response("sentence_parse", result, {"sentence_type": "hypothesis", "order": i})

        # 2. Parse existing premise sentences
        for i, sent in enumerate(premise, 1):
            if sent and sent.strip() and sent.strip().lower() != 'none':
                logger.info(f"Parsing {len(premise)} existing premise(s)")
                logger.debug(f"Existing Premise {i}: {sent[:50]}...")
                result = self._get_parsing_single(sent, "Existing Premise", i)
                if result:
                    results.append(("Existing Premise", i, result))
                    log_llm_response("sentence_parse", result, {"sentence_type": "existing_premise", "order": i})

        # 3. Parse generated premise sentences last
        gen_sentences = generated_premises
        logger.info(f"Parsing {len(gen_sentences)} generated premise(s)")
        for i, sent in enumerate(gen_sentences, 1):
            logger.debug(f"Generated Premise {i}: {sent[:50]}...")
            result = self._get_parsing_single(sent, "Generated Premise", i)
            if result:
                results.append(("Generated Premise", i, result))
                log_llm_response("sentence_parse", result, {"sentence_type": "generated_premise", "order": i})

        if not results:
            logger.error("No sentences were successfully parsed")
            return None, []

        # Aggregate results into expected string format
        aggregated = self._aggregate_parsing_results(results)
        return aggregated, results

    def _get_davidsonian_form(self, parsing_results: list,
                              premise: str,
                              logical_proposition: str,
                              initial_predicates: dict = None,
                              initial_definitions: dict = None) -> str:
        """
        Convert parsed sentences to Davidsonian form, one by one.

        Processing order (determined by parsing_results) is:
        scheme predicates -> hypothesis -> existing premise -> generated premises.
        This ensures generated premises are conditioned on the fixed vocabulary.

        Args:
            parsing_results: List of (sentence_type, order, parsed_text) tuples
            premise: Original premise string (for reference)
            logical_proposition: Logical propositions (for reference)
            initial_predicates: Seed predicates (e.g. from argument scheme consts)

        Returns:
            Aggregated Davidsonian form string
        """
        davidsonian_results = []
        accumulated_predicates = dict(initial_predicates) if initial_predicates else {}
        accumulated_definitions = dict(initial_definitions) if initial_definitions else {}

        logger.info(f"Converting {len(parsing_results)} parsed sentence(s) to Davidsonian form")

        for sentence_type, order, parsed_text in parsing_results:
            if parsed_text == "none":
                davidsonian_results.append((sentence_type, order, "Logical form: none"))
                continue

            # Format accumulated predicates for the prompt (with definitions)
            if accumulated_predicates:
                pred_lines = []
                for name, sig in accumulated_predicates.items():
                    defn = accumulated_definitions.get(name, '')
                    if defn:
                        pred_lines.append(f'  {name} :: "{sig}" -- "{defn}"')
                    else:
                        pred_lines.append(f'  {name} :: "{sig}"')
                existing_predicates_text = (
                    "Previously generated predicates (reuse these where applicable):\n"
                    + '\n'.join(pred_lines)
                )
            else:
                existing_predicates_text = ''

            logger.debug(f"Processing {sentence_type} {order} with {len(accumulated_predicates)} existing predicates")
            result = self._get_davidsonian_single(parsed_text, sentence_type,
                                                  existing_predicates=existing_predicates_text)

            if result:
                # Clean up the result
                cleaned = self._clean_davidsonian_result(result)
                davidsonian_results.append((sentence_type, order, cleaned))

                # Extract typed predicates (with definitions), fall back to regex extraction
                new_preds, new_defs = self._extract_typed_predicates(cleaned)
                if not new_preds:
                    new_preds = self._extract_predicates_from_logical_form(cleaned)
                for name, sig in new_preds.items():
                    if name not in accumulated_predicates:
                        accumulated_predicates[name] = sig
                    else:
                        existing_arity = accumulated_predicates[name].count('=>')
                        new_arity = sig.count('=>')
                        if new_arity > existing_arity:
                            accumulated_predicates[name] = sig
                # Merge definitions (don't overwrite existing)
                for name, defn in new_defs.items():
                    if name not in accumulated_definitions:
                        accumulated_definitions[name] = defn

        # Aggregate into expected format
        return self._aggregate_davidsonian_results(davidsonian_results)

    def _clean_davidsonian_result(self, result: str) -> str:
        """Clean up a single Davidsonian form result."""
        # Remove "Answer:" prefix if present
        result = re.sub(r'^.*?answer:\s*', '', result, flags=re.DOTALL | re.IGNORECASE)

        # Filter to keep only relevant lines
        lines = result.split('\n')
        filtered_lines = []
        keep_line = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Keep lines that are part of the structure
            # Check if line starts with a number followed by a dot (e.g., "1.", "10.", "123.")
            is_numbered = bool(re.match(r'^\d+\.', stripped))
            if (is_numbered or
                stripped.lower().startswith(('subject', 'verb', 'direct', 'linking',
                                             'logical form', 'adverbial', 'prepositional',
                                             'subject complement', 'auxiliary', 'main verb',
                                             '-', 'object', 'predicates'))):
                filtered_lines.append(stripped)
            # Also keep lines that start with sentence type headers
            elif any(stripped.lower().startswith(t) for t in
                     ['hypothesis', 'generated', 'premise', 'existing']):
                filtered_lines.append(stripped)

        return '\n'.join(filtered_lines)

    def _get_axioms(self, davidsonian_form: str,
                    scheme_axiom: str = None, scheme_consts: dict = None,
                    scheme_name: str = None) -> str:
        """
        Generate Isabelle axioms from Davidsonian form automatically.

        Extracts predicates and their types from logical forms, then converts
        the logical forms directly to Isabelle axiom syntax without LLM calls.

        Args:
            davidsonian_form: The aggregated Davidsonian form string
            scheme_axiom: Pre-loaded argument scheme axiom string
            scheme_consts: Pre-loaded argument scheme consts dict
            scheme_name: Name of the argument scheme (for comments)

        Returns:
            Complete Isabelle code with axioms
        """
        # Extract predicates automatically from the logical forms
        logger.info("Extracting predicates from logical forms")
        auto_extracted_predicates = self._extract_predicates_from_davidsonian(davidsonian_form)
        logger.debug(f"Found {len(auto_extracted_predicates)} predicates: {list(auto_extracted_predicates.keys())}")

        # Generate axioms automatically from logical forms
        logger.info("Generating axioms from logical forms")
        axiom_blocks = self._generate_axioms_from_logical_forms(davidsonian_form)
        logger.debug(f"Generated {len(axiom_blocks)} axiom(s)")

        # Merge into single Isabelle code block
        return self._merge_axiom_code(
            auto_extracted_predicates,
            axiom_blocks,
            scheme_axiom=scheme_axiom,
            scheme_consts=scheme_consts,
            scheme_name=scheme_name
        )

    def _get_theorem(self, davidsonian_form: str, axiom: str,
                     premise: str) -> str:
        """
        Generate the theorem systematically without LLM.

        Structure:
        - Existing premises become assumptions
        - Hypothesis becomes the conclusion (shows)
        - Proof uses apply-style: apply (rule scheme), apply auto, using axioms, sledgehammer
        """
        # Extract logical forms
        logical_forms = self._extract_logical_forms_by_section(davidsonian_form)

        # Get hypothesis logical form for 'shows'
        hypothesis_lf = None
        if logical_forms['hypothesis']:
            hypothesis_lf = logical_forms['hypothesis'][0][1]
            hypothesis_lf = self._convert_logical_form_to_isabelle(hypothesis_lf)
            hypothesis_lf = self._add_missing_quantifiers(hypothesis_lf)

        # Get existing premise logical forms for 'assumes' (all of them)
        premise_lfs = []
        asm_labels = []
        if logical_forms['existing_premises']:
            for idx, (order, lf) in enumerate(logical_forms['existing_premises']):
                if lf.lower() != 'none':
                    converted = self._convert_logical_form_to_isabelle(lf)
                    if '-->' in converted:
                        converted = self._add_missing_quantifiers(converted)
                    premise_lfs.append(converted)
                    label = 'asm' if idx == 0 else f'asm_{idx + 1}'
                    asm_labels.append(label)

        # Build the theorem section
        theorem_parts = ['theorem hypothesis:']

        for idx, (label, plf) in enumerate(zip(asm_labels, premise_lfs)):
            if idx == 0:
                theorem_parts.append(f'  assumes {label}: "{plf}"')
            else:
                theorem_parts.append(f'  and {label}: "{plf}"')

        if hypothesis_lf:
            theorem_parts.append(f'  shows "{hypothesis_lf}"')
        else:
            theorem_parts.append('  shows "True"')

        # Get scheme rule name from current_scheme (set in _get_axioms)
        scheme_rule = None
        if hasattr(self, 'current_scheme') and self.current_scheme:
            scheme_rule = self.current_scheme.get('rule_name')

        # Get list of generated premise axiom names
        num_premises = len(logical_forms['generated_premises'])
        axiom_names = [f'generated_premise_{i}' for i in range(1, num_premises + 1)]

        if scheme_rule:
            # --- Isar proof from scheme ---
            # 1. Parse scheme → premises, metavars, conclusion
            parsed = self._parse_scheme_premises(self.current_scheme)

            # 2. Generate obtain step from first existential assumption
            obtain_info = None
            obtain_asm_label = None
            for idx, plf in enumerate(premise_lfs):
                info = self._generate_obtain_step(plf)
                if info:
                    obtain_asm_label = asm_labels[idx]
                    # Update obtain step to use the correct asm label
                    info['step'] = info['step'].replace(
                        'using asm ', f'using {obtain_asm_label} ')
                    info['asm_label'] = obtain_asm_label
                    obtain_info = info
                    break

            # 3. Collect generated premise formulas for LLM context
            gp_formulas = []
            for order, lf in logical_forms['generated_premises']:
                isabelle_lf = self._convert_logical_form_to_isabelle(lf)
                isabelle_lf = self._add_missing_quantifiers(isabelle_lf)
                gp_formulas.append((f'generated_premise_{order}', isabelle_lf))

            # 4. LLM provides metavar bindings dictionary
            # Pass all assumptions so the LLM has full context
            all_assumptions_text = '\n'.join(
                f'{label}: "{plf}"'
                for label, plf in zip(asm_labels, premise_lfs)
            ) if premise_lfs else None
            bindings = self._get_metavar_bindings(
                parsed, hypothesis_lf, all_assumptions_text,
                obtain_info, gp_formulas)
            self._last_metavar_bindings = bindings

            # 5. Algorithmically substitute bindings into scheme premises
            instantiated = self._apply_metavar_bindings(
                parsed['premises'], bindings)

            # 6. Generate Isar proof from instantiated premises
            isar_proof = self._generate_isar_proof(
                parsed, instantiated, axiom_names, obtain_info,
                has_assumption=bool(premise_lfs),
                asm_labels=asm_labels,
                shows_clause=hypothesis_lf)

            theorem_parts.append(isar_proof)
        else:
            # No scheme - use Isar skeleton so LLM can fill in the proof
            theorem_parts.append('proof -')
            theorem_parts.append('')
            theorem_parts.append('qed')

        # Store axiom_names for later use
        self._pending_axiom_names = axiom_names

        theorem_section = '\n'.join(theorem_parts)
        logger.debug(f"Theorem section to insert:\n{theorem_section}")

        # Replace the stub theorem in axiom code
        if not axiom:
            logger.error("Axiom code is empty or None!")
            return ""

        # Check if the pattern exists in axiom
        # Match either Isar-style (proof-...qed) or apply-style (...oops) theorem blocks
        pattern = r'theorem hypothesis:.*?(?:qed|oops)'
        if not re.search(pattern, axiom, flags=re.DOTALL):
            logger.error(f"Pattern 'theorem hypothesis:...qed/oops' not found in axiom code")
            logger.debug(f"Axiom code:\n{axiom}")

        result = re.sub(
            pattern,
            lambda _: theorem_section,
            axiom,
            flags=re.DOTALL
        )

        logger.debug(f"Result after regex substitution: length={len(result)}")
        return result

    def complete_proof_after_scheme(self, isabelle_code: str) -> str:
        """
        Complete the proof after apply (rule ...) succeeds.
        Replaces oops with: using axioms apply auto, sledgehammer, oops

        Handles both:
        - apply (rule scheme)\n  oops  (no bridge)
        - apply (rule bridge)\n  apply (rule scheme)\n  oops  (with bridge)
        """
        axiom_names = getattr(self, '_pending_axiom_names', [])

        # Build the completion (apply-style, no qed needed)
        completion_parts = []
        if axiom_names:
            for name in axiom_names:
                completion_parts.append(f'  apply (insert {name})')
        completion_parts.append('  sledgehammer')
        completion_parts.append('  oops')
        completion = '\n'.join(completion_parts)

        # Replace oops (after any apply rules) with the full completion
        # This handles both single apply and chained apply (bridge + scheme)
        result = re.sub(r'\n\s*oops', lambda _, c=completion: '\n' + c, isabelle_code, count=1)

        logger.debug(f"Completed proof after scheme rule:\n{result[-500:]}")
        return result

    def _get_logical_proposition(self, generated_premises: str) -> str:
        """
        Extract logical propositions from generated premises, one at a time.

        Args:
            generated_premises: String containing all generated premises

        Returns:
            Aggregated logical propositions and relations
        """
        sentences = self._split_sentences(generated_premises)

        if not sentences:
            return "Logical Propositions:\n\nLogical Relations:\nNone logical relations"

        logger.info(f"Extracting logical propositions from {len(sentences)} premise(s)")

        results = []
        for i, sent in enumerate(sentences, 1):
            logger.debug(f"Proposition {i}: {sent[:50]}...")
            result = self._get_logical_proposition_single(sent, i)
            if result:
                results.append(result)

        if not results:
            return "Logical Propositions:\n\nLogical Relations:\nNone logical relations"

        # Aggregate logical propositions and relations
        return self._aggregate_logical_propositions(results)

    def _process_logical_proposition(self, logical_proposition: str) -> str:

        def parse_input(input_text):
            lines = input_text.strip().split('\n')
            logical_props = {}
            logical_exprs = []
            mode = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('Logical Propositions:'):
                    mode = 'propositions'
                    continue
                elif line.startswith('Logical Relations:'):
                    mode = 'relations'
                    continue
                else:
                    if mode == 'propositions':
                        if ':' in line:
                            key, value = line.split(':', 1)
                            value = value.strip()
                            logical_props[key.strip()] = value
                    elif mode == 'relations':
                        if '#' in line:
                            expr, comment = line.split('#', 1)
                        else:
                            expr = line
                        expr = expr.strip()
                        if expr:  
                            logical_exprs.append(expr)
            return logical_props, logical_exprs

        logical_props, logical_exprs = parse_input(logical_proposition)
        
        if not logical_exprs or all(not expr for expr in logical_exprs):
            result = 'Logical Propositions:\n'
            for key, value in logical_props.items():
                result += f'{key}: {value}\n'
            result += '\nLogical Relations:\nNone logical relations'
            return result

        symbols_dict = {}
        symbol_meanings = {}
        sanitized_symbol_names = {}

        for key, value in logical_props.items():
            value_no_parentheses = re.sub(r'\(.*?\)', '', value).strip()
            key_sanitized = re.sub(r'\W|^(?=\d)', '_', key)
            sanitized_symbol_names[key] = key_sanitized
            sym = symbols(key_sanitized)
            symbols_dict[key_sanitized] = sym
            symbol_meanings[sym] = value_no_parentheses  

        propositions = []
        initial_implications_set = set()

        for expr in logical_exprs:
            if not expr:  
                continue
            expr_replaced = expr
            for original_key, sanitized_key in sanitized_symbol_names.items():
                pattern = r'\b{}\b'.format(re.escape(original_key))
                expr_replaced = re.sub(pattern, sanitized_key, expr_replaced)
            local_dict = {
                **symbols_dict,
                "Not": Not,
                "And": And,
                "Or": Or,
                "Implies": Implies,
                "Equivalent": Equivalent,
            }
            try:
                proposition = eval(expr_replaced, {"__builtins__": {}}, local_dict)
                if proposition is not None:  
                    propositions.append(proposition)
                    proposition = sympy.simplify(proposition)
                    initial_implications_set.add(proposition)
            except Exception as e:
                continue  

        derived_implications = set()

        def is_equivalent(expr1, expr2):
            return not satisfiable(Not(Equivalent(expr1, expr2)))

        def is_entailed(propositions, conclusion):
            premises = And(*propositions)
            formula = Implies(premises, conclusion)
            return not satisfiable(Not(formula))

        logical_atoms = set()
        for prop in propositions:
            logical_atoms.update(prop.atoms())

        all_literals = set()
        for atom in logical_atoms:
            all_literals.add(atom)
            all_literals.add(Not(atom))

        possible_pairs = product(all_literals, repeat=2)

        for antecedent, consequent in possible_pairs:
            if antecedent == consequent:
                continue  
            conclusion = Implies(antecedent, consequent)
            if any(is_equivalent(conclusion, imp) for imp in initial_implications_set):
                continue  
            if is_entailed(propositions, conclusion):
                derived_implications.add(conclusion)

        def expr_to_meaning(expr):
            if isinstance(expr, sympy.Symbol):
                return symbol_meanings.get(expr, str(expr))
            elif expr.func == Not:
                return f'Not({expr_to_meaning(expr.args[0])})'
            elif expr.func == Implies:
                return f'Implies({expr_to_meaning(expr.args[0])}, {expr_to_meaning(expr.args[1])})'
            elif expr.func == Equivalent:
                return f'Equivalent({expr_to_meaning(expr.args[0])}, {expr_to_meaning(expr.args[1])})'
            else:
                return str(expr)

        def implication_expr_to_str(expr):
            if expr.func == Not:
                arg = implication_expr_to_str(expr.args[0])
                return f'Not({arg})'
            elif expr.func == Implies:
                ant = implication_expr_to_str(expr.args[0])
                con = implication_expr_to_str(expr.args[1])
                return f'Implies({ant}, {con})'
            elif expr.func == Equivalent:
                ant = implication_expr_to_str(expr.args[0])
                con = implication_expr_to_str(expr.args[1])
                return f'Equivalent({ant}, {con})'
            elif isinstance(expr, sympy.Symbol):
                for original_key, sanitized_key in sanitized_symbol_names.items():
                    if str(expr) == sanitized_key:
                        return original_key
                return str(expr)
            else:
                return str(expr)

        result = ''

        result += 'Logical Propositions:\n'
        for key, meaning in logical_props.items():
            result += f'{key}: {meaning}\n'

        result += '\nLogical Relations:\n'
        for prop_expr, prop in zip(logical_exprs, propositions):
            prop_meaning = expr_to_meaning(prop)
            result += f'{prop_expr}\n{prop_meaning}\n--------\n'

        result += '\nDerived Implications:\n'
        for implication in derived_implications:
            implication_str = implication_expr_to_str(implication)
            implication_meaning = expr_to_meaning(implication)
            result += f'{implication_str}\n{implication_meaning}\n--------\n'

        return result

    def get_isabelle_proof(self, premise: str, generated_premises: str,
                           hypothesis: str, isabelle_code: str) -> str:
        logical_proposition = self.logical_proposition
        logical_information = self._process_logical_proposition(logical_proposition)
        lines = isabelle_code.split('\n')
        known_information = ""
        try_to_prove = ""
        for line in lines:
            if line.strip().startswith("assumes asm:"):
                known_information = line.split('"')[1]
            elif line.strip().startswith("shows"):
                try_to_prove = line.split('"')[1]
        # print(derived_rules)
        for _ in range(5):
            inference_result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name=self.prompt_dict['get isabelle proof'],
                premise=premise,
                generated_premises=generated_premises,
                hypothesis=hypothesis,
                isabelle_code=isabelle_code,
                known_information=known_information,
                try_to_prove=try_to_prove,
                logical_information=logical_information
            )
            if 'proof -' in inference_result and 'qed' in inference_result:
                proof_content_pattern = r'proof -.*?qed'
                match = re.search(proof_content_pattern, inference_result,
                                  re.DOTALL)
                if match:
                    inference_result = match.group(0)
                    lines = inference_result.split('\n')
                    modified_lines = []
                    for line in lines:
                        if 'using' in line and '(*' not in line:
                            modi_line = re.sub(r'(.*)\s+using\s+.*?(?=\s*$)',
                                               r'\1 <ATP>',
                                               line, flags=re.DOTALL)
                            modified_lines.append(modi_line)
                        else:
                            modified_lines.append(line)
                    inference_result = '\n'.join(modified_lines)
                    proof_pattern = r'proof -.*?qed'
                    isabelle_code = re.sub(proof_pattern, lambda _: inference_result,
                                           isabelle_code, flags=re.DOTALL)
                    isabelle_code = self._fix_assume_quantifier(isabelle_code)
                return isabelle_code, logical_information
        return isabelle_code, logical_information

    def _load_walton_schemes(self) -> list[dict]:
        """Load Walton argumentation schemes from config."""
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        schemes_config = config.get('walton_argumentation_schemes', {})
        if not schemes_config.get('enabled', True):
            return []

        return schemes_config.get('schemes', [])

    def _inject_scheme_axioms(self, isabelle_code: str, schemes_to_use: list[str]) -> str:
        """
        Inject selected Walton scheme axioms into Isabelle theory.

        Args:
            isabelle_code: Current Isabelle theory code
            schemes_to_use: List of scheme names to inject

        Returns:
            Modified Isabelle code with scheme axioms added
        """
        all_schemes = self._load_walton_schemes()

        # Find the schemes to inject
        schemes_to_inject = [s for s in all_schemes if s['name'] in schemes_to_use]

        if not schemes_to_inject:
            return isabelle_code

        # Find insertion point (after explanation axioms, before theorem)
        lines = isabelle_code.split('\n')
        insert_index = None

        for i, line in enumerate(lines):
            if line.strip().startswith('theorem hypothesis:'):
                insert_index = i
                break

        if insert_index is None:
            return isabelle_code

        # Build scheme axiom block
        scheme_block = ["\n(* Walton Argumentation Schemes *)"]
        for scheme in schemes_to_inject:
            scheme_block.append(f"(* {scheme['name']}: {scheme['description']} *)")
            # Extract axiom text (remove leading/trailing whitespace)
            axiom_text = scheme['isabelle_axiom'].strip()
            scheme_block.append(axiom_text)
            scheme_block.append("")

        # Insert before theorem
        lines.insert(insert_index, '\n'.join(scheme_block))

        return '\n'.join(lines)

    def _format_schemes_for_prompt(self, schemes: list[dict]) -> str:
        """Format Walton schemes for LLM prompt context."""
        if not schemes:
            return "No Walton argumentation schemes available."

        formatted = "Available Walton Argumentation Schemes:\n\n"
        for i, scheme in enumerate(schemes, 1):
            formatted += f"{i}. **{scheme['name']}**\n"
            formatted += f"   Description: {scheme['description']}\n"
            formatted += f"   Axiom:\n"
            # Indent the axiom for readability
            axiom_lines = scheme['isabelle_axiom'].strip().split('\n')
            for line in axiom_lines:
                formatted += f"   {line}\n"
            formatted += "\n"

        return formatted

    def get_logical_form(self, premise: str, generated_premises: str,
                         hypothesis: str, logical_form: str) -> str:
        inference_result = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['get logical form'],
            premise=premise,
            generated_premises=generated_premises,
            hypothesis=hypothesis,
            logical_form=logical_form
        )
        return inference_result

    def _reset_isar_tactics(self, code: str) -> str:
        """Replace all tactics in Isar have/show steps with sorry.

        After LLM syntax refinement the proof may contain real tactics
        (by blast, by simp, etc.) instead of sorry placeholders.  This
        resets them so the Isar step loop can solve them properly.

        Obtain steps (using asm by blast) are left untouched.
        Apply-style proofs (no proof-/qed block) are left untouched.
        """
        proof_match = re.search(
            r'(proof\b[^\n]*)(.*?)(qed)', code, re.DOTALL)
        if not proof_match:
            return code  # No Isar proof block — nothing to reset

        proof_body = proof_match.group(2)
        lines = proof_body.split('\n')
        new_lines = []

        for line in lines:
            stripped = line.strip()

            # Standalone tactic line (by blast, by simp, by (metis ...), etc.)
            # Reset to sorry — obtain step tactics are treated the same as
            # have/show tactics; the sledgehammer loop will re-solve them.
            if re.match(r'by\s+', stripped):
                # Replace with sorry at same indentation
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + 'sorry')
                continue

            # Inline tactic on a using line: "using facts by blast"
            # or "using facts by (metis ...)"
            inline_match = re.search(
                r'\bby\s+(?:\w+(?:\s*\([^)]*\))?|\([^)]*\))', stripped)
            if inline_match and ('have ' in stripped or 'show ' in stripped
                                 or 'using ' in stripped):
                # Remove inline tactic, add sorry on next line
                cleaned = re.sub(
                    r'\s*\bby\s+(?:\w+(?:\s*\([^)]*\))?|\([^)]*\))\s*$',
                    '', line).rstrip()
                indent = len(line) - len(line.lstrip())
                new_lines.append(cleaned)
                new_lines.append(' ' * indent + 'sorry')
                continue

            new_lines.append(line)

        new_proof_body = '\n'.join(new_lines)
        # Normalize show?thesis → show ?thesis (LLM sometimes drops the space)
        new_proof_body = re.sub(r'\bshow\s*\?thesis\b', 'show ?thesis', new_proof_body)
        result = (code[:proof_match.start()]
                  + proof_match.group(1)
                  + new_proof_body
                  + proof_match.group(3)
                  + code[proof_match.end():])

        if new_proof_body != proof_body:
            logger.info("Reset Isar proof tactics to sorry after syntax refinement")

        return result

    def fix_inner_syntax_error(self, isabelle_code: str,
                               error_detail: str, inner_code: str) -> str:
        refined_code = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['refine inner syntax error'],
            code=isabelle_code,
            error_detail=error_detail,
            code_cause_error=inner_code
        )
        refined_code = self._add_quotes(refined_code)
        refined_code = self._fix_unicode_symbols(refined_code)
        refined_code = self._fix_c_style_application(refined_code)
        refined_code = self._fix_assume_quantifier(refined_code)
        refined_code = self._reset_isar_tactics(refined_code)
        return refined_code

    def fix_contradiction_error(self, isabelle_code: str,
                                contradiction_code: str) -> str:
        refined_code = self.llm.generate(
            model_prompt_dir='formalisation_model',
            prompt_name=self.prompt_dict['refine contradiction'],
            natural_language=self.logical_form,
            code=isabelle_code,
            code_cause_error=contradiction_code
        )
        refined_code = self._add_quotes(refined_code)
        refined_code = self._fix_unicode_symbols(refined_code)
        refined_code = self._fix_c_style_application(refined_code)
        refined_code = self._fix_assume_quantifier(refined_code)
        refined_code = self._reset_isar_tactics(refined_code)
        return refined_code

    def formalise(self, theory_name: str, premise: list,
                  generated_premises: list, hypothesis: str,
                  logical_form: str = 'event-based semantics',
                  argumentation_scheme: str = None) -> str:
        self._pending_axiom_names = []  # Clear stale state from previous iterations
        # Store NL inputs so binding/rebinding calls can include them as context
        self._nl_premise = premise
        self._nl_hypothesis = hypothesis

        if logical_form == 'event-based semantics':
            logger.info("Starting sentence-by-sentence processing")

            # Load argument scheme early so its predicates seed the davidsonian form
            scheme_axiom = None
            scheme_consts = None
            scheme_definitions = None
            self.current_scheme = None

            if argumentation_scheme:
                logger.info(f"Loading argument scheme: {argumentation_scheme}")
                schemes = self._load_walton_schemes()
                scheme = next((s for s in schemes if s['name'] == argumentation_scheme), None)
                if scheme:
                    scheme_axiom = scheme.get('isabelle_axiom')
                    raw_consts = scheme.get('consts', {})
                    # Handle both old format (Name: "type") and new format (Name: {type: "type", definition: "defn"})
                    scheme_consts = {}
                    scheme_definitions = {}
                    for name, value in raw_consts.items():
                        if isinstance(value, dict):
                            scheme_consts[name] = value.get('type', "'a => bool")
                            defn = value.get('definition', '')
                            if defn:
                                scheme_definitions[name] = defn
                        else:
                            scheme_consts[name] = value
                    self.current_scheme = scheme
                    logger.info(f"Loaded scheme with {len(scheme_consts)} consts, {len(scheme_definitions)} definitions")
                else:
                    logger.warning(f"Scheme '{argumentation_scheme}' not found in config")

            # Step 1: Parse sentences one by one
            logger.info("Step 1: Syntactic Parsing (sentence-by-sentence)")
            syntactic_parsed_structure, parsing_results = self._get_parsing(
                premise,
                generated_premises,
                hypothesis
            )

            if syntactic_parsed_structure is None:
                logger.error("Syntactic parsing failed")
                return None

            # Step 2: Get logical propositions (sentence-by-sentence)
            logger.info("Step 2: Extracting Logical Propositions (sentence-by-sentence)")
            self.logical_proposition = self._get_logical_proposition(generated_premises)

            # Step 3: Convert to Davidsonian form (sentence-by-sentence)
            # Scheme predicates seed the accumulator, then hypothesis, existing
            # premise, and finally generated premises (conditioned on all above).
            logger.info("Step 3: Davidsonian Form Conversion (sentence-by-sentence)")
            davidsonian_form = self._get_davidsonian_form(
                parsing_results,
                premise,
                self.logical_proposition,
                initial_predicates=scheme_consts,
                initial_definitions=scheme_definitions
            )

            logger.debug(f"Syntactic Parsing Result:\n{syntactic_parsed_structure}")
            logger.debug(f"Neodavidsonian Form:\n{davidsonian_form}")

            # Step 4: Generate axioms (sentence-by-sentence)
            logger.info("Step 4: Generating Isabelle Axioms (sentence-by-sentence)")
            axioms = self._get_axioms(
                davidsonian_form,
                scheme_axiom=scheme_axiom,
                scheme_consts=scheme_consts,
                scheme_name=argumentation_scheme
            )
            logger.debug(f"Axioms code length: {len(axioms) if axioms else 0}")
            logger.debug(f"Axioms code:\n{axioms[:500] if axioms else 'EMPTY'}...")

            if not axioms or len(axioms.strip()) < 50:
                logger.error(f"Axioms generation failed or returned empty result")

            # Step 5: Generate theorem (this step remains batched as it needs full context)
            logger.info("Step 5: Generating Theorem")
            self.logical_form = davidsonian_form  # needed by _get_metavar_bindings
            theorem = self._get_theorem(davidsonian_form, axioms, premise)
            logger.debug(f"Theorem code length: {len(theorem) if theorem else 0}")
            logger.debug(f"Theorem code:\n{theorem[:500] if theorem else 'EMPTY'}...")

            if not theorem or len(theorem.strip()) < 50:
                logger.error(f"Theorem generation failed or returned empty result")

            # Sanitize theory name for Isabelle compatibility (no hyphens allowed)
            sanitized_name = self._sanitize_theory_name(theory_name)
            self.code = f'theory {sanitized_name}\n' + theorem
            self.code = self._fix_unicode_symbols(self.code)
            self.code = self._fix_c_style_application(self.code)
            logger.debug(f"Final code length: {len(self.code)}")
            # Don't call _clean_proof - we generate proof structure systematically now
            self.save_formalised_kb(self.code, sanitized_name)

            logger.info("Formalisation processing complete")
        return self.code

    def _sanitize_theory_name(self, theory_name: str) -> str:
        """
        Sanitize theory name for Isabelle compatibility.
        Isabelle theory names cannot contain hyphens - only underscores and alphanumerics.
        """
        # Replace hyphens with underscores
        sanitized = theory_name.replace('-', '_')
        # Remove any other invalid characters (keep only alphanumeric and underscore)
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', sanitized)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized

    # ==================== Bridge Axiom Methods (Abductive Instantiation) ====================

    def _extract_domain_conclusion(self, isabelle_code: str) -> str:
        """Extract the 'shows' statement from Isabelle code."""
        match = re.search(r'shows\s+"([^"]+)"', isabelle_code)
        return match.group(1) if match else ""

    def _build_predicate_defs_text(self) -> str:
        """Return scheme predicate definitions + logical form predicates as a formatted string."""
        pred_lines = []
        scheme_consts = {}
        if hasattr(self, 'current_scheme') and self.current_scheme:
            raw = self.current_scheme.get('consts', {})
            for name, val in raw.items():
                type_sig = val.get('type', val) if isinstance(val, dict) else val
                defn = val.get('definition', '') if isinstance(val, dict) else ''
                scheme_consts[name] = type_sig
                if defn:
                    pred_lines.append(f'{name} :: "{type_sig}" -- "{defn}"')
                else:
                    pred_lines.append(f'{name} :: "{type_sig}"')
        if hasattr(self, 'logical_form') and self.logical_form:
            preds, defs = self._extract_typed_predicates(self.logical_form)
            for name, sig in preds.items():
                if name in scheme_consts:
                    continue
                defn = defs.get(name, '')
                if defn:
                    pred_lines.append(f'{name} :: "{sig}" -- "{defn}"')
                else:
                    pred_lines.append(f'{name} :: "{sig}"')
        return '\n'.join(pred_lines) if pred_lines else '(none)'

    def _extract_existing_consts(self, isabelle_code: str) -> str:
        """Extract existing const declarations from Isabelle code as text.

        Returns a formatted string of all const type signatures with definitions
        where available (from self.const_definitions), e.g.:
          Predicate :: "agent => bool" -- "Predicate(x) means ..."
          Relation :: "agent => action => bool"
        """
        lines = []
        in_consts = False
        for line in isabelle_code.splitlines():
            stripped = line.strip()
            if stripped.startswith('consts'):
                in_consts = True
                continue
            if in_consts:
                if '::' in stripped and stripped:
                    name_match = re.match(r'(\w+)\s*::', stripped)
                    if name_match:
                        name = name_match.group(1)
                        defn = self.const_definitions.get(name)
                        lines.append(stripped + (f' -- "{defn}"' if defn else ''))
                    else:
                        lines.append(stripped)
                elif stripped == '':
                    continue
                else:
                    in_consts = False
        return '\n'.join(lines) if lines else '(none)'

    def _get_bridge_axioms(self, scheme: dict, isabelle_code: str, previous_attempts: list = None) -> str:
        """
        Generate bridge axioms so the proof's show clause can be derived
        from the available premises and generated axioms.

        Provides the model with all available facts (assumes + generated
        axioms) and the exact show clause so it can produce a concrete,
        grounded bridge without needing to reason about scheme structure.
        """
        # Extract the exact show clause (goal that must be proved)
        show_clause = self._extract_domain_conclusion(isabelle_code)
        if not show_clause:
            logger.warning("Could not extract show clause for bridge axioms")
            return ""

        # Collect available premises: assumes clause + generated axiom formulas
        # Handle both `assumes "..."` and `assumes label: "..."` forms
        assumes_match = re.search(r'assumes\s+(?:\w+:\s*)"([^"]+)"', isabelle_code)
        assumed_premise = (f'"{assumes_match.group(1)}"'
                           if assumes_match else '(none)')

        generated_axioms = self.extract_generated_axioms(isabelle_code)
        gen_text = ('\n'.join(f'{a["name"]}: "{a["formula"]}"'
                              for a in generated_axioms)
                    if generated_axioms else '(none)')

        available_premises = (
            f'Assumed premise:\n{assumed_premise}\n\nGenerated axioms:\n{gen_text}'
        )

        existing_consts = self._extract_existing_consts(isabelle_code)

        predicate_defs_text = self._build_predicate_defs_text()

        logger.info(f"Generating bridge axioms: {len(generated_axioms)} generated axioms "
                     f"==> {show_clause[:50]}...")

        if previous_attempts:
            prev_text = '\n\n'.join(
                f'Attempt {j+1}:\n{a}' for j, a in enumerate(previous_attempts)
            )
            previous_attempts_text = (
                f"PREVIOUS BRIDGE ATTEMPTS (all failed — do NOT regenerate these):\n{prev_text}"
            )
        else:
            previous_attempts_text = ""

        try:
            result = self.llm.generate(
                model_prompt_dir='formalisation_model',
                prompt_name=self.prompt_dict.get('get bridge axioms',
                                                  'get_bridge_axioms_prompt.txt'),
                available_premises=available_premises,
                show_clause=show_clause,
                existing_types=existing_consts,
                predicate_definitions=predicate_defs_text,
                previous_attempts=previous_attempts_text
            )

            cleaned = self._clean_bridge_axioms(result, isabelle_code)
            if cleaned:
                logger.debug(f"Generated bridge axioms:\n{cleaned}")
            return cleaned

        except Exception as e:
            logger.error(f"Error generating bridge axioms: {e}")
            return ""

    def _object_to_metalevel(self, formula: str) -> str:
        """Convert the outermost object-level --> to metalevel ==>.

        Only the first --> is converted. Everything after it is kept as-is
        (the conclusion), which may itself contain --> for complex propositions
        that need to unify as a single term with scheme variables.

        A --> B             becomes  A ==> B
        A --> B --> C       becomes  A ==> B --> C   (B --> C is the conclusion)
        A \\<and> B --> C   becomes  [| A; B |] ==> C
        """
        # Already metalevel — nothing to do
        if '==>' in formula or '[|' in formula:
            return formula

        if '-->' not in formula:
            return formula

        # Split on the FIRST --> only: premises --> conclusion (conclusion kept as-is)
        match = re.match(r'(.+?)\s*-->\s*(.+)', formula, re.DOTALL)
        if not match:
            return formula

        premises_str = match.group(1).strip()
        conclusion = match.group(2).strip()

        # Split premises on \<and> (Isabelle object-level conjunction)
        premises = re.split(r'\\<and>', premises_str)
        premises = [p.strip() for p in premises if p.strip()]

        if len(premises) == 1:
            return f"{premises[0]} ==> {conclusion}"
        else:
            joined = '; '.join(premises)
            return f"[| {joined} |] ==> {conclusion}"

    def _clean_bridge_axioms(self, raw_result: str, isabelle_code: str = '') -> str:
        """Extract valid axiomatization block from LLM output.

        Takes the FIRST code block only (the LLM's initial answer is best;
        later blocks tend to be tautological fallbacks). If the first block
        contains a raw formula rather than a full axiomatization block, wraps
        it automatically.

        Strips any const declarations that already exist in the theory to avoid
        type conflicts (e.g., LLM redeclaring agent => bool as 'a => bool).
        """
        if not raw_result:
            return ""

        # Take the FIRST code block only — ignore any subsequent blocks
        code_match = re.search(r'```(?:\w*)\n(.*?)```', raw_result, re.DOTALL)
        content = code_match.group(1).strip() if code_match else raw_result.strip()

        # If the block is a raw formula (no axiomatization keyword), wrap it
        if content and 'axiomatization' not in content:
            formula = content.strip().strip('"')
            # Discard tautologies: "A --> A"
            if '-->' in formula:
                lhs, _, rhs = formula.partition('-->')
                if lhs.strip() == rhs.strip():
                    logger.debug("Bridge axiom is a tautology — discarding")
                    return ""
            content = f'axiomatization where\n  bridge_axiom: "{formula}"'

        # Collect existing const names from the theory
        existing_const_names = set()
        if isabelle_code:
            in_consts = False
            for line in isabelle_code.splitlines():
                stripped = line.strip()
                if stripped.startswith('consts'):
                    in_consts = True
                    continue
                if in_consts:
                    if '::' in stripped:
                        name = stripped.split('::')[0].strip()
                        existing_const_names.add(name)
                    elif stripped == '':
                        continue
                    else:
                        in_consts = False

        # Extract consts and axiomatization blocks
        parts = []
        consts_match = re.search(r'consts\s.*?(?=\n\n|\naxiomatization|$)',
                                 content, re.DOTALL)
        if consts_match:
            consts_block = consts_match.group(0).strip()
            # Filter out const declarations that already exist in the theory
            if existing_const_names:
                filtered_lines = ['consts']
                for line in consts_block.splitlines():
                    stripped = line.strip()
                    if '::' in stripped:
                        name = stripped.split('::')[0].strip()
                        if name not in existing_const_names:
                            filtered_lines.append(line)
                    elif stripped != 'consts':
                        filtered_lines.append(line)
                # Only include consts block if there are new consts
                if len(filtered_lines) > 1:
                    parts.append('\n'.join(filtered_lines))
            else:
                parts.append(consts_block)

        axiom_match = re.search(r'axiomatization\s+where.*?(?=\n\n|\ntheorem|\nend|$)',
                                content, re.DOTALL)
        if axiom_match:
            axiom_block = axiom_match.group(0).strip()
            parts.append(axiom_block)

        return "\n\n".join(parts) if parts else ""

    def _inject_bridge_axioms(self, isabelle_code: str, bridge_axioms: str) -> tuple:
        """
        Inject bridge axioms before the theorem statement and wire them into
        the Isar show ?thesis using clause so sledgehammer can find them.

        New consts introduced by bridge chaining are merged into the main
        consts block (so they persist across bridge retries). Only the
        axiomatization block lives in the replaceable bridge section.

        Returns (updated_isabelle_code, new_const_names).
        """
        if not bridge_axioms:
            return isabelle_code, []

        # Split bridge output into new consts, axiom-only block, and definitions
        new_consts, axiom_only, new_definitions = self._parse_axiom_code(bridge_axioms)
        self.const_definitions.update(new_definitions)
        if not axiom_only:
            axiom_only = bridge_axioms.strip()

        # Extract all bridge axiom rule names (first after 'where', rest after 'and')
        bridge_rule_names = re.findall(
            r'(?:axiomatization\s+where\s*\n?\s*|\band\s+)(\w+)\s*:',
            axiom_only
        )

        # Inject new consts into the main consts block so they persist across retries
        new_const_names = []
        if new_consts:
            existing_names = {m.group(1) for m in re.finditer(r'^\s+(\w+)\s*::', isabelle_code, re.MULTILINE)}
            fresh = {k: v for k, v in new_consts.items() if k not in existing_names}
            if fresh:
                new_const_names = list(fresh.keys())
                new_const_lines = '\n'.join(f'  {n} :: "{t}"' for n, t in fresh.items())
                isabelle_code = re.sub(
                    r'(consts\b.*?\n)((?:  \S.*\n)*)',
                    lambda m: m.group(0) + new_const_lines + '\n',
                    isabelle_code, count=1, flags=re.DOTALL
                )
                logger.info(f"Merged new bridge consts into main block: {new_const_names}")

        # Replace existing bridge axiomatization block or insert before theorem
        existing_bridge = re.search(
            r'\n\(\* Bridge axioms for scheme instantiation \*\)\n.*?(?=\n\(|\ntheorem\s)',
            isabelle_code, re.DOTALL
        )
        if existing_bridge:
            isabelle_code = (
                isabelle_code[:existing_bridge.start()] +
                f"\n\n(* Bridge axioms for scheme instantiation *)\n{axiom_only}\n" +
                isabelle_code[existing_bridge.end():]
            )
        else:
            match = re.search(r'\ntheorem\s+', isabelle_code)
            if match:
                pos = match.start()
                isabelle_code = (isabelle_code[:pos] +
                        f"\n\n(* Bridge axioms for scheme instantiation *)\n{axiom_only}\n" +
                        isabelle_code[pos:])

        # For Isar proofs: add bridge names to the 'using' clause of show ?thesis
        # so sledgehammer has them as explicit hints
        if bridge_rule_names:
            bridge_names_str = ' '.join(bridge_rule_names)
            isar_show = re.search(
                r'(show\s+\?thesis\s*\n\s*using\s+)([^\n]+)(\s*\n\s*sorry)',
                isabelle_code
            )
            if isar_show:
                new_using = isar_show.group(2).strip() + ' ' + bridge_names_str
                isabelle_code = (isabelle_code[:isar_show.start()] +
                                 isar_show.group(1) + new_using +
                                 isar_show.group(3) +
                                 isabelle_code[isar_show.end():])
                logger.info(f"Added bridge axioms '{bridge_names_str}' to show ?thesis using clause")

        return self._fix_unicode_symbols(isabelle_code), new_const_names

    def save_formalised_kb(self, isabelle_code: str, theory_name: str) -> None:
        # Sanitize theory name for Isabelle compatibility
        sanitized_name = self._sanitize_theory_name(theory_name)

        # Check if code already starts with theory declaration
        if not isabelle_code.strip().startswith('theory '):
            # Only add theory header if not already present
            isabelle_code = re.sub(r'.*imports Main', 'imports Main',
                                   isabelle_code, flags=re.DOTALL)
            isabelle_code = f'theory {sanitized_name}\n' + isabelle_code
        else:
            # Replace existing theory name with sanitized version
            isabelle_code = re.sub(
                r'^theory\s+\S+',
                f'theory {sanitized_name}',
                isabelle_code
            )

        self.code = isabelle_code
        # Use sanitized name for filename too
        with open(f'{self.isabelle_dir}/{sanitized_name}.thy', 'w') as f:
            f.write(isabelle_code)

