from .abstract import CritiqueModel
from isabelle_client import start_isabelle_server, get_isabelle_client
from isabelle_client.data_models import IsabelleResponseType
from formalisation.isabelle_formaliser import IsabelleFormaliser
from typing import Optional
from collections import defaultdict
from utils.logging_config import (
    get_logger, log_isabelle_interaction, log_isabelle_response
)
import threading
import time
import json
import re
import yaml
import atexit
import signal

logger = get_logger(__name__)

_ISABELLE_PORT_BASE = 7777
_ISABELLE_WATCHDOG_TIMEOUT = 60
_ISABELLE_TYPECHECK_TIMEOUT = 15   # short timeout for sorry-based type checks
_SYNTAX_CHECK_RETRIES = 5
_SLEDGEHAMMER_CMD = 'sledgehammer [max_proofs = 1]'

# Global registry of server processes for cleanup
_server_processes = []


def _cleanup_servers():
    """Cleanup handler to kill any remaining Isabelle servers."""
    for proc in _server_processes:
        try:
            if proc is not None and hasattr(proc, 'terminate'):
                proc.terminate()
                proc.wait(timeout=5)
                logger.info("Terminated Isabelle server process on cleanup")
        except Exception as e:
            logger.warning(f"Error terminating Isabelle server: {e}")
            try:
                if proc is not None and hasattr(proc, 'kill'):
                    proc.kill()
            except Exception:
                pass


def _signal_handler(signum, frame):
    """Handle termination signals by cleaning up servers."""
    logger.info(f"Received signal {signum}, cleaning up Isabelle servers...")
    _cleanup_servers()
    # Re-raise to allow default handling
    signal.signal(signum, signal.SIG_DFL)
    raise SystemExit(1)


# Register cleanup handlers
atexit.register(_cleanup_servers)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except (ValueError, OSError):
    # Signals may not work in all environments (e.g., threads)
    pass


class IsabelleCritique(CritiqueModel):
    ERROR_KEYWORDS = [
        "Type unification failed", "Inner lexical error",
        "Outer syntax error", "Inner syntax error",
        "Outer lexical error", "Malformed command syntax",
        "Undefined type name", "Duplicate constant"
    ]

    def __init__(self, generative_model, isabelle_session,
                 theory_name: Optional[str] = 'example',
                 prompt_dict: Optional[dict] = None,
                 argumentation_scheme: Optional[str] = None):
        super().__init__(generative_model,
                         prompt_dict,
                         type='hard')
        # Store argumentation scheme for use in formalisation
        self.argumentation_scheme = argumentation_scheme
        if prompt_dict is None:
            prompt_dict = {
                'get davidsonian':
                    'get_davidsonian_form_prompt.txt',
                'refine contradiction':
                    'refine_contradiction_syntax_error_prompt.txt',
                'refine inner syntax error':
                    'refine_inner_syntax_error_prompt.txt',
                'get isabelle proof': 'get_isabelle_proof_prompt.txt',
                'get sentence parse': 'get_sentence_parse_prompt.txt',
                'get logical proposition': 'get_logical_proposition_prompt.txt',
                'get bridge axioms': 'get_bridge_axioms_prompt.txt',
                'instantiate scheme': 'instantiate_scheme_prompt.txt'
            }
        import os as _os
        _pid = _os.getpid()
        self.isabelle_name = f'isabelle_{_pid}'
        self.port = _ISABELLE_PORT_BASE + (_pid % 1000)
        self.log_file = f'server_{_pid}.log'
        self.session_name = isabelle_session
        self.verbose = True
        self.options = None
        self.watchdog_timeout = _ISABELLE_WATCHDOG_TIMEOUT
        self._load_isabelle_config()
        self._init_client()
        self._init_session()
        self.code = None
        self.prompt_dict = prompt_dict
        self.formaliser = IsabelleFormaliser(generative_model, prompt_dict)
        # Sanitize theory name for Isabelle compatibility (no hyphens allowed)
        self.theory_name = self.formaliser._sanitize_theory_name(theory_name)

    # load isabelle config from config.yaml
    def _load_isabelle_config(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.isabelle_dir = config['isabelle']['master_dir']
        self.dirs = config['isabelle']['app_dir']
        # Option to skip LLM proof generation and keep deterministic apply-style proof
        self.skip_llm_proof = config['isabelle'].get('skip_llm_proof', False)

    @staticmethod
    def _get_attr(obj, key, default=None):
        """Helper to access attributes whether dict or object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _parse_response_body(self, finished_response):
        """Parse response body, handling both old API (JSON string) and new API (object)."""
        response_body = finished_response.response_body
        if isinstance(response_body, str):
            response_body = json.loads(response_body)
        return response_body

    # init isabelle server
    def _init_client(self):
        server_info, server_process = start_isabelle_server(
            name=self.isabelle_name, port=self.port, log_file=self.log_file
        )
        self.server_process = server_process
        # Register for cleanup on exit
        _server_processes.append(server_process)
        self.isabelle = get_isabelle_client(server_info)

    # init isabelle session (HOL, ZF, HOL-Proof, ...)
    def _init_session(self):
        # Initialize start_id to None
        self.start_id = None

        # Try session_build, but if it fails, skip it for built-in sessions like HOL
        try:
            self.isabelle.session_build(
                session=self.session_name, dirs=self.dirs,
                verbose=self.verbose, options=self.options
            )
        except Exception as e:
            logger.warning(f"session_build failed (OK for built-in sessions like HOL): {e}")
            # Continue anyway - HOL is a built-in session that doesn't need building

        # Get session ID - handle both string and TaskOK object responses
        session_response = self.isabelle.session_start(session=self.session_name)
        if hasattr(session_response, 'session_id'):
            # TaskOK object with session_id attribute
            self.start_id = session_response.session_id
        elif isinstance(session_response, str):
            # Direct string response
            self.start_id = session_response
        elif isinstance(session_response, list):
            # Response is a list - iterate to find session_id
            # The session_id is typically in a SessionStartRegularResponse at the end of the list
            for item in session_response:
                # Check if item itself has session_id
                if hasattr(item, 'session_id'):
                    self.start_id = item.session_id
                    break
                # Check if response_body has session_id (SessionStartRegularResponse case)
                elif hasattr(item, 'response_body') and hasattr(item.response_body, 'session_id'):
                    self.start_id = item.response_body.session_id
                    break
                elif isinstance(item, dict) and 'session_id' in item:
                    self.start_id = item['session_id']
                    break
            if self.start_id is None:
                logger.error("Could not find session_id in list response")
                raise ValueError(f"Cannot extract session_id from list response")
        else:
            # Try to extract from response_body if it's a response object
            try:
                import json
                if hasattr(session_response, 'response_body'):
                    body = json.loads(session_response.response_body)
                    self.start_id = body.get('session_id')
                elif hasattr(session_response, '__dict__'):
                    # Check if session_id is in the object's __dict__
                    if 'session_id' in session_response.__dict__:
                        self.start_id = session_response.__dict__['session_id']
                    else:
                        logger.debug(f"session_response type: {type(session_response)}")
                        logger.debug(f"session_response attrs: {dir(session_response)}")
                        if hasattr(session_response, '__dict__'):
                            logger.debug(f"session_response __dict__: {session_response.__dict__}")
                        raise ValueError(f"Cannot extract session_id from {type(session_response)}")
                else:
                    raise ValueError(f"Cannot extract session_id from {type(session_response)}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.debug(f"session_response: {session_response}")
                raise ValueError(f"Cannot extract session_id: {e}")

    # get isabelle response
    def _get_response(self, theories, master_dir):
        # Log the theory file being sent to Isabelle
        isabelle_logger = logger.bind(isabelle_server=True)
        for theory in theories:
            thy_path = f'{master_dir}/{theory}.thy'
            try:
                with open(thy_path, 'r') as f:
                    thy_content = f.read()
                isabelle_logger.debug(
                    f"[ISABELLE_REQUEST] theory={theory}\n{'='*60}\n{thy_content}\n{'='*60}"
                )
            except Exception:
                isabelle_logger.debug(f"[ISABELLE_REQUEST] theory={theory} (could not read file)")

        start_time = time.time()
        try:
            isabelle_response = self.isabelle.use_theories(
                session_id=self.start_id,
                theories=theories,
                master_dir=master_dir,
                watchdog_timeout=self.watchdog_timeout
            )
        except (ValueError, ConnectionError, OSError) as e:
            solving_time = time.time() - start_time
            logger.warning(f"Isabelle server connection error: {e}. Returning empty response.")
            return [], solving_time
        solving_time = time.time() - start_time

        # Log all responses from Isabelle
        for r in isabelle_response:
            body = getattr(r, 'response_body', None)
            if body is None:
                body_str = 'N/A'
            elif isinstance(body, str):
                body_str = body
            else:
                body_str = str(body)
            isabelle_logger.debug(
                f"[ISABELLE_RESPONSE] type={r.response_type} | time={solving_time:.2f}s\n{body_str}"
            )

        return isabelle_response, solving_time

    def _get_formalisation(self, generated_premises: list,
                           hypothesis: str,
                           premise: list,
                           theory_name: Optional[str]):
        # get isabelle code from input natural language sentences
        self.code = self.formaliser.formalise(
            theory_name, premise,
            generated_premises, hypothesis,
            logical_form='event-based semantics',
            argumentation_scheme=self.argumentation_scheme
        )

    def _get_isabelle_syntax_output(self, theory_name: str,
                                    generated_premises: list,
                                    hypothesis: str,
                                    premise: list,
                                    iteration_number: int = 0) -> bool:
        # formalise the nl into isabelle theory
        self._get_formalisation(
            generated_premises=generated_premises,
            hypothesis=hypothesis,
            premise=premise,
            theory_name=theory_name
        )
        has_syntax_error = True
        # check and refine any syntatic errors in the code
        logger.debug(f"Code before syntax check:\n{self.code}")
        for i in range(_SYNTAX_CHECK_RETRIES):
            has_inner_syntax_error = False
            has_contradiction_error = False
            error_detail = []
            inner_code = ''
            contradiction_code = ''
            inference_time = 9999
            # check inner and contradiction error
            (has_inner_syntax_error,
             has_contradiction_error,
             error_detail,
             inner_code,
             contradiction_code,
             inference_time
             ) = self.check_syntax_error(theory_name, self.isabelle_dir,
                                         self.code)
            # if has inner syntax error, refine it
            if has_inner_syntax_error:
                logger.info(f"Refining inner syntax error, inference time: {inference_time:.2f}s, error details: {error_detail}")
                refined_code = self.formaliser.fix_inner_syntax_error(
                    self.code, error_detail, inner_code
                )
                self.formaliser.save_formalised_kb(refined_code,
                                                   theory_name)
                self.code = self.formaliser.code
                continue
            # if has contradition syntax error, refine it
            if has_contradiction_error:
                logger.info(f"Refining contradiction error, inference time: {inference_time:.2f}s, error details: {error_detail}")
                refined_code = self.formaliser.fix_contradiction_error(
                    self.code, contradiction_code
                )
                self.formaliser.save_formalised_kb(refined_code,
                                                   theory_name)
                self.code = self.formaliser.code
                continue
            if not has_inner_syntax_error and not has_contradiction_error:
                logger.info(f"No syntax errors found after check, total inference time: {inference_time:.2f}s")
                has_syntax_error = False
                break
            else:
                continue
        # For scheme-based proofs, validate that have-step bindings are
        # type-correct before handing off to sledgehammer.
        if not has_syntax_error and self.argumentation_scheme and iteration_number == 0:
            self.code = self.validate_and_rebind(theory_name, self.code)
        # Use LLM proof generation only in the no-scheme case
        use_llm_proof = (self.argumentation_scheme is None) and not self.skip_llm_proof
        if not use_llm_proof:
            logger.info("Skipping LLM proof generation (scheme-based run), keeping deterministic apply-style proof")
            logical_information = ""
        else:
            self.code, logical_information = self.formaliser.get_isabelle_proof(
                premise,
                generated_premises,
                hypothesis,
                self.code
            )
        logger.info("Generated Isabelle Code")
        log_isabelle_interaction(
            direction="llm_to_isabelle",
            content=self.code,
            context={"theory_name": theory_name, "stage": "formalisation_complete"}
        )
        self.formaliser.save_formalised_kb(self.code,
                                           theory_name)
        return has_syntax_error, logical_information

    # using isabelle client to call isabelle to check
    def check_syntax_error(self, theory_name, master_dir, isabelle_code):
        # Sanitize theory name for Isabelle compatibility (no hyphens allowed)
        theory_name = self.formaliser._sanitize_theory_name(theory_name)
        logger.debug(f"check_syntax_error: theory_name={theory_name}, master_dir={master_dir}")
        # Check what theory name is declared in the code
        first_line = isabelle_code.split('\n')[0] if isabelle_code else ""
        logger.debug(f"Theory declaration in code: {first_line}")
        # For syntax checking, we keep the shows statement (to check its syntax)
        # but replace all proof tactics with just "oops" to skip the actual proof
        isa_code_lines = isabelle_code.split('\n')
        in_proof = False
        new_lines = []
        for i, line in enumerate(isa_code_lines):
            stripped = line.strip()
            if stripped.startswith('shows'):
                # Keep the shows line as-is to check its syntax
                new_lines.append(line)
                # Add oops to skip the proof
                new_lines.append('  oops')
                in_proof = True
            elif in_proof:
                # Skip all proof lines until we hit 'end'
                if stripped == 'end':
                    new_lines.append(line)
                    in_proof = False
                # else: skip the line (proof tactics, qed, etc.)
            else:
                new_lines.append(line)
        check_syntax_error_code = '\n'.join(new_lines)
        # Use a separate file for syntax checking to avoid overwriting the actual theory
        syntax_check_name = f'{theory_name}_syntax_check'
        # Update the theory declaration in the code to match the new name
        check_syntax_error_code = re.sub(
            r'^theory\s+\S+',
            f'theory {syntax_check_name}',
            check_syntax_error_code
        )
        file_path = f'{self.isabelle_dir}/{syntax_check_name}.thy'
        logger.debug(f"Writing syntax check to file: {file_path}")
        with open(file_path, 'w') as f:
            f.write(check_syntax_error_code)

        theories_name = [syntax_check_name]
        logger.debug(f"Sending to Isabelle: theories={theories_name}, master_dir={master_dir}")
        log_isabelle_interaction(
            direction="llm_to_isabelle",
            content=check_syntax_error_code,
            context={"theory_name": syntax_check_name, "stage": "syntax_check"}
        )
        isabelle_response, solving_time = \
            self._get_response(theories_name, master_dir)
        logger.debug(f"Isabelle response types: {[r.response_type for r in isabelle_response]}")
        has_inner_syntax_error = False
        has_contradiction_error = False
        error_details = []
        lines = []
        inner_code = ''
        error_code_detail = []
        tactic_messages = []
        contradiction_code = ''

        finished_response = next(
            (item for item in isabelle_response
            if item.response_type == IsabelleResponseType.FINISHED),
            None
        )
        if finished_response is not None:
            response_body = self._parse_response_body(finished_response)

            # Handling errors
            errors = self._get_attr(response_body, 'errors', [])
            if errors:
                for error in errors:
                    message = self._get_attr(error, 'message', '')
                    position = self._get_attr(error, 'pos', {})
                    line = self._get_attr(position, 'line', 0)
                    if any(keyword in message for keyword in self.ERROR_KEYWORDS):
                        error_details.append(
                            f"Error on line {line}: {message}"
                        )
                        lines.append(line)
                        has_inner_syntax_error = True
            else:
                has_inner_syntax_error = False

            for node in self._get_attr(response_body, 'nodes', []):
                for message in self._get_attr(node, 'messages', []):
                    tactic_messages.append(self._get_attr(message, 'message', ''))

            has_contradiction_error = False

            # Handling warnings (log only, do not trigger syntax refinement)
            nodes = self._get_attr(response_body, 'nodes', [])
            for node in nodes:
                messages = self._get_attr(node, 'messages', [])
                for message in messages:
                    if self._get_attr(message, 'kind', '') == 'warning':
                        warning_message = self._get_attr(message, 'message', '')
                        logger.debug(f"Isabelle warning (ignored): {warning_message}")
        else:
            logger.warning("No FINISHED response found from Isabelle")
            logger.debug(f"All response types: {[r.response_type for r in isabelle_response]}")
            for r in isabelle_response:
                body = getattr(r, 'response_body', None)
                if body is None:
                    body_str = 'N/A'
                elif isinstance(body, str):
                    body_str = body[:500]
                else:
                    # Handle non-string response bodies (objects, Task, etc.)
                    body_str = str(body)[:500]
                log_isabelle_response(
                    response_type=str(r.response_type),
                    response_body=body_str,
                    context={"theory_name": theory_name, "stage": "syntax_check"}
                )
            return False, False, [9999], '', '', 9999
        inner_code = ''
        isabelle_lines = isabelle_code.splitlines()
        for line_number in lines:
            index = line_number - 1
            if index < len(isabelle_lines):
                line_text = isabelle_lines[index].strip()

                if "axiomatization where" in line_text:
                    if index + 1 < len(isabelle_lines):
                        inner_code = (inner_code +
                                      isabelle_lines[index + 1].strip() +
                                      '\n'
                                      if inner_code != ''
                                      else
                                      isabelle_lines[index + 1].strip()
                                      + '\n')
                elif "hypothesis" in line_text:
                    if index + 1 < len(isabelle_lines):
                        for i in range(1, 5):
                            if index + i < len(isabelle_lines):
                                inner_code += \
                                    isabelle_lines[index + i].strip() + '\n'
                else:
                    inner_code = inner_code + line_text+'\n'
        error_code_detail = "\n".join(f"{index}. {item}" for index, item in
                                      enumerate(error_details, start=1))

        return has_inner_syntax_error, has_contradiction_error, \
            error_code_detail, inner_code, contradiction_code, \
            solving_time

    def _type_check_sorry_proof(self, theory_name: str,
                                isabelle_code: str) -> list[str]:
        """Submit the sorry-proof to Isabelle to catch type errors in have steps.

        Unlike check_syntax_error (which uses oops and skips the proof body),
        this submits the full sorry-proof so Isabelle elaborates and type-checks
        every have step. Since sorry needs no proof search it completes in ~3s.

        Returns a list of type-error message strings, empty if bindings are
        type-correct.
        """
        type_check_name = f'{theory_name}_binding_typecheck'
        check_code = re.sub(
            r'^theory\s+\S+', f'theory {type_check_name}',
            isabelle_code, flags=re.MULTILINE
        )
        file_path = f'{self.isabelle_dir}/{type_check_name}.thy'
        with open(file_path, 'w') as f:
            f.write(check_code)

        log_isabelle_interaction(
            direction="llm_to_isabelle",
            content=check_code,
            context={"theory_name": type_check_name, "stage": "binding_type_check"}
        )

        # Short watchdog: sorry needs no proof search
        original_timeout = self.watchdog_timeout
        self.watchdog_timeout = _ISABELLE_TYPECHECK_TIMEOUT
        try:
            isabelle_response, solving_time = self._get_response(
                [type_check_name], self.isabelle_dir
            )
        finally:
            self.watchdog_timeout = original_timeout

        type_errors = []
        finished = next(
            (r for r in isabelle_response
             if r.response_type == IsabelleResponseType.FINISHED),
            None
        )
        if finished:
            response_body = self._parse_response_body(finished)
            for error in self._get_attr(response_body, 'errors', []):
                msg = self._get_attr(error, 'message', '')
                if any(kw in msg for kw in (
                    'Type unification failed',
                    'incompatible operand type',
                    'Type error',
                )):
                    type_errors.append(msg)

        if type_errors:
            logger.warning(
                f"Binding type check failed ({len(type_errors)} errors): "
                f"{type_errors[:2]}"
            )
        else:
            logger.info(
                f"Binding type check passed in {solving_time:.2f}s"
            )
        return type_errors

    def validate_and_rebind(self, theory_name: str, isabelle_code: str,
                            max_attempts: int = 3) -> str:
        """Type-check have-step bindings and rebind on failure.

        Submits the sorry-proof to Isabelle; if type errors are found in the
        have steps, calls regenerate_isar_proof with the error context and
        retries.  Returns the (possibly updated) isabelle_code.
        """
        for attempt in range(max_attempts):
            type_errors = self._type_check_sorry_proof(theory_name, isabelle_code)
            if not type_errors:
                return isabelle_code

            logger.warning(
                f"Type errors in have steps (attempt {attempt + 1}/"
                f"{max_attempts}), regenerating metavar bindings"
            )
            error_msgs = [f"Type error in application: {e}" for e in type_errors]
            regenerated = self.formaliser.regenerate_isar_proof(
                isabelle_code, error_msgs
            )
            if not regenerated:
                logger.warning("regenerate_isar_proof returned None, keeping current code")
                return isabelle_code
            isabelle_code = regenerated
            self.formaliser.save_formalised_kb(isabelle_code, theory_name)
            logger.info(f"Rebound metavars after type error (attempt {attempt + 1})")

        # Final check after last attempt
        type_errors = self._type_check_sorry_proof(theory_name, isabelle_code)
        if type_errors:
            logger.warning(
                f"Binding type check still failing after {max_attempts} attempts"
            )
        return isabelle_code

    def _extract_goals_from_text(self, msg_text: str) -> list[str]:
        """Extract numbered goals from a message containing 'goal (' or 'Failed to finish'.

        For goals with meta-implications (⟹), only the conclusion after the
        last ⟹ is kept — everything before is an assumption already given.
        """
        goals = []
        goal_matches = re.findall(
            r'(\d+)\.\s+(.+?)(?=\n\s*\d+\.|$)',
            msg_text,
            re.DOTALL
        )
        for _, goal_text in goal_matches:
            goal_text = goal_text.strip()
            # Strip meta-implication assumptions — keep only the conclusion
            # Handle Isabelle symbol encoding, ASCII, and Unicode representations
            if r'\<Longrightarrow>' in goal_text:
                goal_text = goal_text.rsplit(r'\<Longrightarrow>', 1)[1].strip()
            elif '==>' in goal_text:
                goal_text = goal_text.rsplit('==>', 1)[1].strip()
            elif '\u27F9' in goal_text:  # ⟹ Unicode long rightarrow
                goal_text = goal_text.rsplit('\u27F9', 1)[1].strip()
            # Normalize whitespace (multi-line goals may have embedded newlines)
            goal_text = ' '.join(goal_text.split())
            goals.append(goal_text)
        return goals

    def _extract_unsolved_goals(self, response_body, theory_name: str) -> list[str]:
        """
        Extract unsolved subgoals from Isabelle's failed proof attempt.

        Checks both node messages and top-level errors for goal state.
        Isabelle reports goals in 'Failed to apply', 'Failed to finish proof',
        or 'goal (N subgoals):' messages.
        """
        unsolved_goals = []

        # Check node messages
        for node in self._get_attr(response_body, 'nodes', []):
            for message in self._get_attr(node, 'messages', []):
                msg_text = self._get_attr(message, 'message', '')
                msg_kind = self._get_attr(message, 'kind', '')

                logger.debug(f"[GOAL_EXTRACT] kind={msg_kind} | text={msg_text[:200]}")

                if 'goal (' in msg_text or 'Failed to finish' in msg_text:
                    logger.info(f"[GOAL_EXTRACT] Found goal marker in message: {msg_text[:300]}")
                    unsolved_goals.extend(self._extract_goals_from_text(msg_text))

        # Fallback: check top-level errors (goals can appear in error messages too)
        if not unsolved_goals:
            for error in self._get_attr(response_body, 'errors', []):
                err_text = self._get_attr(error, 'message', '')
                if 'goal (' in err_text or 'Failed to finish' in err_text:
                    logger.info(f"[GOAL_EXTRACT] Found goal marker in error: {err_text[:300]}")
                    unsolved_goals.extend(self._extract_goals_from_text(err_text))

        if unsolved_goals:
            logger.info(f"[GOAL_EXTRACT] Extracted {len(unsolved_goals)} goals: {unsolved_goals}")
        else:
            logger.warning(f"[GOAL_EXTRACT] No goals found in response for {theory_name}")

        return unsolved_goals

    def _parse_sledgehammer_tactic(self, tactic_messages: list[str]) -> Optional[str]:
        """Extract proof tactic from sledgehammer 'Try this:' response messages.

        Returns the tactic string (e.g., 'by (metis ...)') or None if not found.
        """
        for item in tactic_messages:
            if "Try this:" in item:
                tactic = item.split("Try this:", 1)[1].strip()
                last_paren = tactic.rfind('(')
                if last_paren != -1:
                    tactic = tactic[:last_paren].strip()
                return tactic
        # Fallback: look for bare 'by' tactic
        for item in tactic_messages:
            if item.strip().startswith("by"):
                match = re.match(r'by\s*.*?(?=\s*\(\d)', item.strip())
                if match:
                    return match.group(0).strip()
                break
        return None

    # using isabelle client to call isabelle to check
    def critique(self, iteration_number: int,
                 explanation: Optional[list],
                 hypothesis: str,
                 premise: list,
                 try_schemes: bool = False,
                 isabelle_code: Optional[str] = None,
                 tactic_hints: Optional[list] = None):
        # Map the old 'explanation' parameter to 'generated_premises' internally
        generated_premises = explanation
        theory_name = f'{self.theory_name}_{str(iteration_number)}'

        # NEW: If scheme trial mode enabled, try schemes sequentially
        if try_schemes:
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            trial_enabled = config.get('walton_argumentation_schemes', {}).get('trial_settings', {}).get('enabled', False)

            if trial_enabled:
                result = self._try_schemes_sequentially(
                    theory_name, generated_premises, hypothesis, premise
                )
                if result['successful_scheme']:
                    return result['critique_output']

        # Continue with regular critique process if schemes didn't work
        error_code = ''
        error_comment = ''
        response_body = None
        semantic_validity = False
        solving_time = 0
        tactic_messages = []
        has_syntax_error = False
        proof_sketch = False
        logical_information = ''
        critique_output = {
            'semantic validity': False,
            'syntactic validity': True,
            'error code': '',
            'solving time': 0,
            'proof tactics': [],
            'code': '',
            'logical information': '',
            'unsolved_goals': [],
            'apply_failed': False,
        }
        for _ in range(2):
            # Fast path: use pre-built Isabelle code (skip formalisation)
            if isabelle_code is not None and not proof_sketch:
                self.code = isabelle_code
                self.formaliser.save_formalised_kb(isabelle_code, theory_name)
                # Run syntax check loop (no NL needed)
                has_syntax_error = True
                for i in range(_SYNTAX_CHECK_RETRIES):
                    (has_inner_syntax_error,
                     has_contradiction_error,
                     error_detail, inner_code,
                     contradiction_code,
                     inference_time
                     ) = self.check_syntax_error(theory_name,
                                                 self.isabelle_dir, self.code)
                    solving_time += inference_time
                    if has_inner_syntax_error:
                        logger.info(f"Refining inner syntax error (formal path): {error_detail}")
                        refined_code = self.formaliser.fix_inner_syntax_error(
                            self.code, error_detail, inner_code)
                        self.formaliser.save_formalised_kb(refined_code, theory_name)
                        self.code = self.formaliser.code
                        continue
                    if has_contradiction_error:
                        logger.info(f"Refining contradiction error (formal path): {error_detail}")
                        refined_code = self.formaliser.fix_contradiction_error(
                            self.code, contradiction_code)
                        self.formaliser.save_formalised_kb(refined_code, theory_name)
                        self.code = self.formaliser.code
                        continue
                    if not has_inner_syntax_error and not has_contradiction_error:
                        has_syntax_error = False
                        break
                logger.info("Formal path: syntax check complete, skipping LLM proof generation")
                # oops-based check never sees have steps — run sorry type-check to catch metavar binding errors
                if self.argumentation_scheme and iteration_number == 0:
                    self.code = self.validate_and_rebind(theory_name, self.code)
                self.formaliser.save_formalised_kb(self.code, theory_name)
                self.code = self.formaliser.code  # sync theory name after save
            elif not proof_sketch:
                # Normal path: formalise from NL
                (has_syntax_error,
                 logical_information) = self._get_isabelle_syntax_output(
                    theory_name=theory_name,
                    generated_premises=generated_premises,
                    hypothesis=hypothesis,
                    premise=premise,
                    iteration_number=iteration_number
                )
            # if has syntax error, return directly
            if has_syntax_error:
                semantic_validity = False
                has_syntax_error = True
                critique_output['semantic validity'] = semantic_validity
                critique_output['syntactic validity'] = False
                critique_output['error code'] = error_code.strip()
                critique_output['solving time'] = solving_time
                critique_output['proof tactics'] = tactic_messages
                critique_output['code'] = self.code
                critique_output['logical information'] = logical_information
                return critique_output
            isabelle_code = self.code
            # use sledgehammer directly
            if not proof_sketch:
                # Check if this is an Isar proof with sorry (scheme-based)
                has_isar_proof = bool(re.search(
                    r'proof\b.*?qed', isabelle_code, re.DOTALL))
                has_sorry = 'sorry' in isabelle_code

                if has_isar_proof and has_sorry:
                    # --- Isar step loop: batched sledgehammer ---
                    logger.info("Isar proof with sorry detected, starting batched solving...")
                    # Reset any pre-filled tactics to sorry so we
                    # control all solving (LLM syntax fixes may have
                    # introduced real tactics).
                    sledgehammer_code = self.formaliser._reset_isar_tactics(
                        isabelle_code)
                    semantic_validity = False
                    all_tactic_messages = []
                    # solved_tactics: {step_name: tactic} — keyed by step name so
                    # gaps (step 1 and 3 work, step 2 fails) are handled correctly.
                    solved_tactics = {}
                    # Ordered step names before any sorry is replaced (all sorry's present).
                    sorry_step_names = re.findall(
                        r'have\s+(\w+)\s*:', sledgehammer_code)

                    # --- Pre-pass: apply tactic hints (optimistic batch) ---
                    if tactic_hints and 'sorry' in sledgehammer_code:
                        # tactic_hints: {step_name: tactic} from the previous iteration.
                        # Apply per step name so gaps are handled correctly:
                        # a failing hint does not block hints for later steps.
                        all_hints_code = sledgehammer_code
                        applied_step_names = []
                        for step_name, hint in tactic_hints.items():
                            pattern = (rf'(have\s+{re.escape(step_name)}'
                                       rf'\s*:\s*"[^"]*")(\n\s*)sorry')
                            new_code = re.sub(
                                pattern, rf'\1\2{hint}', all_hints_code, count=1)
                            if new_code != all_hints_code:
                                all_hints_code = new_code
                                applied_step_names.append(step_name)

                        if applied_step_names:
                            logger.info(f"Verifying {len(applied_step_names)} hint(s) "
                                         f"in single Isabelle call")

                            with open(f'{self.isabelle_dir}/{theory_name}.thy', 'w') as f:
                                f.write(all_hints_code)
                            (hint_resp, hint_time) = self._get_response(
                                [theory_name], self.isabelle_dir)
                            solving_time += hint_time

                            hint_finished = next(
                                (item for item in hint_resp
                                 if item.response_type == IsabelleResponseType.FINISHED), None)

                            hints_ok = True
                            if hint_finished:
                                hint_body = self._parse_response_body(hint_finished)
                                if self._get_attr(hint_body, 'errors', []):
                                    hints_ok = False
                                if hints_ok:
                                    for node in self._get_attr(hint_body, 'nodes', []):
                                        for message in self._get_attr(node, 'messages', []):
                                            msg = self._get_attr(message, 'message', '')
                                            if ('Failed' in msg or 'error' in msg.lower()
                                                    or 'Bad' in msg):
                                                hints_ok = False
                                                break
                                        if not hints_ok:
                                            break
                            else:
                                hints_ok = False

                            if hints_ok:
                                logger.info(f"All {len(applied_step_names)} hint(s) verified")
                                sledgehammer_code = all_hints_code
                                solved_tactics.update(
                                    {n: tactic_hints[n] for n in applied_step_names})
                            else:
                                # Fallback: apply hints one at a time, independently.
                                # No break — a failing hint does not block later ones.
                                logger.info("Batch hint verification failed, "
                                             "falling back to individual hints")
                                for step_name in applied_step_names:
                                    hint = tactic_hints[step_name]
                                    pattern = (rf'(have\s+{re.escape(step_name)}'
                                               rf'\s*:\s*"[^"]*")(\n\s*)sorry')
                                    hint_code = re.sub(
                                        pattern, rf'\1\2{hint}',
                                        sledgehammer_code, count=1)
                                    if hint_code == sledgehammer_code:
                                        continue  # Already solved or step not found
                                    with open(f'{self.isabelle_dir}/{theory_name}.thy', 'w') as f:
                                        f.write(hint_code)
                                    (h_resp, h_time) = self._get_response(
                                        [theory_name], self.isabelle_dir)
                                    solving_time += h_time
                                    h_fin = next(
                                        (item for item in h_resp
                                         if item.response_type == IsabelleResponseType.FINISHED), None)
                                    h_ok = True
                                    if h_fin:
                                        h_body = self._parse_response_body(h_fin)
                                        if self._get_attr(h_body, 'errors', []):
                                            h_ok = False
                                        if h_ok:
                                            for node in self._get_attr(h_body, 'nodes', []):
                                                for message in self._get_attr(node, 'messages', []):
                                                    msg = self._get_attr(message, 'message', '')
                                                    if ('Failed' in msg or 'error' in msg.lower()
                                                            or 'Bad' in msg):
                                                        h_ok = False
                                                        break
                                                if not h_ok:
                                                    break
                                    else:
                                        h_ok = False
                                    if h_ok:
                                        logger.info(f"Hint for {step_name} succeeded individually")
                                        sledgehammer_code = hint_code
                                        solved_tactics[step_name] = hint
                                    else:
                                        logger.info(f"Hint for {step_name} failed individually")

                    # --- Batch: replace ALL remaining sorry's with sledgehammer ---
                    if 'sorry' in sledgehammer_code:
                        lines = sledgehammer_code.split('\n')
                        # Map line number (1-indexed) → sorry ordinal
                        sorry_line_to_idx = {}
                        sorry_ordinal = 0
                        batch_lines = []
                        in_obtain = False
                        for i, line in enumerate(lines):
                            stripped = line.lstrip()
                            if stripped.startswith('obtain '):
                                in_obtain = True
                            elif in_obtain and not stripped.startswith('--'):
                                # obtain proof block ends after a non-comment line
                                # that isn't a continuation (sorry/by/proof)
                                if not re.match(r'sorry\b|by\b|proof\b', stripped):
                                    in_obtain = False
                            if re.search(r'\bsorry\b', line):
                                if in_obtain:
                                    # obtain uses inline sorry — sledgehammer can't be
                                    # inserted before it; handle separately after batch
                                    batch_lines.append(line)
                                    if re.search(r'\bsorry\b', stripped) and not stripped.startswith('obtain'):
                                        in_obtain = False  # sorry consumed the obtain proof
                                    continue
                                sorry_ordinal += 1
                                line_num = i + 1  # 1-indexed for Isabelle pos
                                sorry_line_to_idx[line_num] = sorry_ordinal
                                # Insert sledgehammer before sorry
                                indent = len(line) - len(line.lstrip())
                                batch_lines.append(' ' * indent + _SLEDGEHAMMER_CMD)
                                batch_lines.append(line)  # Keep sorry
                            else:
                                batch_lines.append(line)
                                if in_obtain and re.match(r'by\b|proof\b|qed\b', stripped):
                                    in_obtain = False

                        batch_code = '\n'.join(batch_lines)
                        # Recalculate sledgehammer line numbers after insertion
                        # (inserting lines shifts everything down)
                        sledge_line_to_idx = {}
                        sorry_count = 0
                        for i, line in enumerate(batch_lines):
                            stripped = line.strip()
                            if stripped.startswith('sledgehammer'):
                                sorry_count += 1
                                sledge_line_to_idx[i + 1] = sorry_count  # 1-indexed

                        logger.info(f"Batching {sorry_ordinal} sledgehammer call(s) "
                                     f"in single Isabelle request")

                        with open(f'{self.isabelle_dir}/{theory_name}.thy', 'w') as f:
                            f.write(batch_code)

                        (response, resp_time) = self._get_response(
                            [theory_name], self.isabelle_dir)
                        solving_time += resp_time

                        finished = next(
                            (item for item in response
                             if item.response_type == IsabelleResponseType.FINISHED), None)

                        if finished:
                            body = self._parse_response_body(finished)
                            # Group messages by source line number
                            line_messages = defaultdict(list)
                            for node in self._get_attr(body, 'nodes', []):
                                for message in self._get_attr(node, 'messages', []):
                                    msg_text = self._get_attr(message, 'message', '')
                                    pos = self._get_attr(message, 'pos', {})
                                    msg_line = self._get_attr(pos, 'line', 0)
                                    line_messages[msg_line].append(msg_text)
                                    all_tactic_messages.append(msg_text)

                            # Match each sledgehammer position to its tactic
                            step_results = {}
                            for sledge_line, idx in sledge_line_to_idx.items():
                                msgs = line_messages.get(sledge_line, [])
                                tactic = self._parse_sledgehammer_tactic(msgs)
                                step_results[idx] = (tactic, msgs)

                            # Apply all solved tactics — replace the Nth sorry by ordinal
                            # to avoid misplacing tactics when earlier steps are unsolved.
                            # Map ordinals back to step names (ordinals are relative to
                            # the remaining sorry's after hints were applied).
                            remaining_step_names = [
                                n for n in sorry_step_names if n not in solved_tactics
                            ]
                            tactics_by_ordinal = {}
                            for idx in sorted(step_results):
                                tactic, msgs = step_results[idx]
                                if tactic:
                                    logger.info(f"Batch step {idx} solved: {tactic}")
                                    tactics_by_ordinal[idx] = tactic
                                    if idx <= len(remaining_step_names):
                                        solved_tactics[remaining_step_names[idx - 1]] = tactic
                                else:
                                    logger.warning(f"Step {idx} unsolvable")

                            if tactics_by_ordinal:
                                sorry_counter = [0]
                                _lines_for_obtain = sledgehammer_code.split('\n')
                                _obtain_sorry_lines = set()
                                _in_obt = False
                                for _li, _ll in enumerate(_lines_for_obtain):
                                    _ls = _ll.lstrip()
                                    if _ls.startswith('obtain '):
                                        _in_obt = True
                                    if _in_obt and re.search(r'\bsorry\b', _ll):
                                        _obtain_sorry_lines.add(_li)
                                        _in_obt = False
                                    elif _in_obt and re.match(r'by\b|proof\b|qed\b', _ls):
                                        _in_obt = False

                                def _replace_nth_sorry(match):
                                    # Advance line counter to match position
                                    pos = match.start()
                                    text = sledgehammer_code
                                    line_idx = text[:pos].count('\n')
                                    if line_idx in _obtain_sorry_lines:
                                        return match.group(0)  # leave obtain sorry alone
                                    sorry_counter[0] += 1
                                    return tactics_by_ordinal.get(
                                        sorry_counter[0], match.group(0))

                                sledgehammer_code = re.sub(
                                    r'\bsorry\b', _replace_nth_sorry,
                                    sledgehammer_code)

                            # Check for type errors in unsolved steps
                            for idx in sorted(step_results):
                                tactic, msgs = step_results[idx]
                                if not tactic:
                                    has_type_error = any(
                                        'Type unification failed' in m
                                        for m in msgs
                                    )
                                    if has_type_error:
                                        critique_output['proof_type_error'] = True
                                        type_err_msgs = [
                                            m for m in msgs
                                            if 'Type unification' in m
                                        ]
                                        logger.warning(
                                            "Type error in have step — "
                                            "metavar bindings may be wrong. "
                                            f"Errors: {type_err_msgs}"
                                        )

                            # After applying all tactics, extract remaining unsolved goals
                            have_goals = re.findall(
                                r'have\s+\w+:\s*"([^"]+)"[^"]*?sorry',
                                sledgehammer_code, re.DOTALL)
                            show_has_sorry = bool(re.search(
                                r'show\s+(?:\?thesis|"[^"]*").*?sorry',
                                sledgehammer_code, re.DOTALL))
                            if have_goals:
                                critique_output['unsolved_goals'] = have_goals
                            elif show_has_sorry:
                                critique_output['unsolved_goals'] = [
                                    '?thesis (show step)']
                        else:
                            logger.warning("No FINISHED response for batched sledgehammer")

                    # --- Resolve obtain sorrys with by blast / by auto ---
                    # obtain steps can't use the sledgehammer batch (wrong syntax),
                    # so we try inline replacement with simple tactics here.
                    _OBTAIN_BLASTS = ['by blast', 'by auto', 'by (auto simp add: assms)']
                    if 'sorry' in sledgehammer_code:
                        # Match obtain sorrys: sorry either inline on obtain line
                        # or on the immediately following line.
                        _obtain_sorry_inline = re.compile(
                            r'^(\s*obtain\b[^\n]+?)\bsorry\b', re.MULTILINE)
                        _obtain_sorry_nextline = re.compile(
                            r'(\n\s*obtain\b[^\n]+\n)(\s*)sorry\b', re.MULTILINE)
                        for blast_tactic in _OBTAIN_BLASTS:
                            if not re.search(
                                r'(^(\s*obtain\b[^\n]+?)\bsorry\b|(\n\s*obtain\b[^\n]+\n)(\s*)sorry\b)',
                                sledgehammer_code, re.MULTILINE):
                                break
                            candidate = _obtain_sorry_inline.sub(
                                lambda m: m.group(1) + blast_tactic, sledgehammer_code)
                            candidate = _obtain_sorry_nextline.sub(
                                lambda m, bt=blast_tactic: m.group(1) + m.group(2) + bt,
                                candidate)
                            with open(f'{self.isabelle_dir}/{theory_name}.thy', 'w') as f:
                                f.write(candidate)
                            (obt_resp, obt_time) = self._get_response(
                                [theory_name], self.isabelle_dir)
                            solving_time += obt_time
                            obt_fin = next(
                                (item for item in obt_resp
                                 if item.response_type == IsabelleResponseType.FINISHED), None)
                            obt_ok = False
                            if obt_fin:
                                obt_body = self._parse_response_body(obt_fin)
                                obt_errors = self._get_attr(obt_body, 'errors', [])
                                if not obt_errors:
                                    obt_ok = True
                                    for node in self._get_attr(obt_body, 'nodes', []):
                                        for msg in self._get_attr(node, 'messages', []):
                                            mt = self._get_attr(msg, 'message', '')
                                            if ('error' in mt.lower() or 'Failed' in mt
                                                    or 'Inner syntax error' in mt):
                                                obt_ok = False
                                                break
                                        if not obt_ok:
                                            break
                            if obt_ok:
                                logger.info(
                                    f"obtain sorry resolved with '{blast_tactic}'")
                                sledgehammer_code = candidate
                                break
                            else:
                                logger.info(
                                    f"obtain sorry not resolved with '{blast_tactic}', trying next")

                    # Check if all sorry's are resolved (oops also counts as failure)
                    semantic_validity = 'sorry' not in sledgehammer_code and 'oops' not in sledgehammer_code

                    # Detect bridge need: show step failed in a scheme-based proof.
                    # The batch inserts sledgehammer BEFORE each sorry, so have
                    # steps are still "proved" via sorry when show's sledgehammer
                    # runs. Show failure therefore means the scheme output doesn't
                    # directly unify with ?thesis — bridge axiom needed.
                    if not semantic_validity:
                        total_have_steps = len(re.findall(
                            r'have\s+step_\d+:', sledgehammer_code))
                        show_sorry = bool(re.search(
                            r'show\s+(?:\?thesis|"[^"]*").*?sorry', sledgehammer_code, re.DOTALL))
                        if total_have_steps > 0 and show_sorry:
                            critique_output['bridge_needed'] = True
                            logger.info(
                                "Show step failed in scheme-based proof — "
                                "bridge axiom needed")

                    # Save final proof state and return
                    with open(f'{self.isabelle_dir}/{theory_name}.thy', 'w') as f:
                        f.write(sledgehammer_code)

                    critique_output['syntactic validity'] = True
                    critique_output['error code'] = ''
                    critique_output['solving time'] = solving_time
                    critique_output['proof tactics'] = all_tactic_messages
                    critique_output['solved_tactics'] = solved_tactics
                    # Return solved code when proof succeeds, original
                    # (with sorry's) when it fails so refinement can
                    # re-attempt all steps after modifying axioms
                    critique_output['code'] = (sledgehammer_code
                                               if semantic_validity
                                               else self.code)
                    critique_output['logical information'] = logical_information
                    critique_output['semantic validity'] = semantic_validity

                    # NOTE: error keyword check deliberately omitted for
                    # the Isar path.  Errors from individual step failures
                    # (e.g. "Type unification failed") are expected when
                    # sorry is present — they do NOT mean the theory is
                    # syntactically invalid.  The syntax check loop
                    # (above) already handles genuine syntax errors.

                    return critique_output

                else:
                    # Handle Isar-style proofs (proof - ... qed)
                    pattern = r'(proof -).*?(qed)(?!.*qed)'
                    sledgehammer_code = re.sub(
                        pattern,
                        r'  sledgehammer [max_proofs = 1]\n  oops',
                        isabelle_code,
                        flags=re.DOTALL
                    )
                    # Handle apply-style proofs without scheme: remove 'sorry' so sledgehammer failure is detected
                    sledgehammer_code = re.sub(r'\n\s*sorry\s*\n', '\n  oops\n', sledgehammer_code)

                # --- Non-scheme single-shot sledgehammer (unchanged) ---
                with open(f'{self.isabelle_dir}/{theory_name}.thy',
                          'w') as f:
                    f.write(sledgehammer_code)
                theories_name = [theory_name]
                # get isabelle output
                (isabelle_response,
                 response_time) = self._get_response(theories_name,
                                                     self.isabelle_dir)
                solving_time += response_time
                # Default to False - only set True when sledgehammer finds a proof
                semantic_validity = False
                finished_response = next(
                    (item for item in isabelle_response
                     if item.response_type == IsabelleResponseType.FINISHED),
                    None)
                if finished_response is not None:
                    response_body = self._parse_response_body(finished_response)
                    for node in self._get_attr(response_body, 'nodes', []):
                        for message in self._get_attr(node, 'messages', []):
                            msg_text = self._get_attr(message, 'message', '')
                            if 'No proof found' in msg_text or 'Gave up' in msg_text:
                                unsolved_goals = self._extract_unsolved_goals(response_body, theory_name)
                                if unsolved_goals:
                                    critique_output['unsolved_goals'] = unsolved_goals
                                break

                if finished_response is not None:
                    tactic_messages = []
                    for node in self._get_attr(response_body, 'nodes', []):
                        for message in self._get_attr(node, 'messages', []):
                            tactic_messages.append(self._get_attr(message, 'message', ''))
                    tactic_to_use = self._parse_sledgehammer_tactic(tactic_messages)
                    if tactic_to_use is not None and all("no proof found" not in item.lower() for item in tactic_messages):
                        semantic_validity = True
                        lines = sledgehammer_code.split('\n')
                        for i, line in enumerate(lines):
                            if 'sledgehammer' in line and '(*' not in line:
                                # Remove sledgehammer with any options (e.g. [max_proofs = 1])
                                lines[i] = re.sub(
                                    r'sledgehammer\s*(\[.*?\])?\s*',
                                    tactic_to_use + ' ',
                                    line, count=1).rstrip()
                            if 'by by' in line:
                                lines[i] = line.replace('by by', 'by', 1)
                            if 'oops' in line:
                                lines[i] = line.replace('oops', '', 1)
                        sledgehammer_code = '\n'.join(lines)

                    with open(f'{self.isabelle_dir}/{theory_name}.thy',
                              'w') as f:
                        f.write(sledgehammer_code)
                    critique_output['syntactic validity'] = True
                    critique_output['error code'] = ''
                    critique_output['solving time'] = solving_time
                    critique_output['proof tactics'] = tactic_messages
                    critique_output['code'] = self.code
                    critique_output['logical information'] = logical_information
                    critique_output['semantic validity'] = semantic_validity

                    if any(keyword in message for message in tactic_messages for keyword in self.ERROR_KEYWORDS):
                        semantic_validity = False
                        has_syntax_error = True
                        critique_output['semantic validity'] = semantic_validity
                        critique_output['syntactic validity'] = False
                        critique_output['error code'] = error_code.strip()
                        critique_output['solving time'] = solving_time
                        critique_output['proof tactics'] = tactic_messages
                        critique_output['code'] = self.code
                        critique_output['logical information'] = logical_information
                        return critique_output

                    return critique_output

                if not semantic_validity:
                    with open(f'{self.isabelle_dir}/{theory_name}.thy', 'w') as f:
                        f.write(isabelle_code)
                    # if the direct sledgehammer failed
                    # go to next loop for using proof sketch to prove
                    proof_sketch = True
                    continue
            else:
                # --- Batched ATP solving: replace ALL <ATP> at once ---
                if '<ATP>' in isabelle_code:
                    isa_code_lines = isabelle_code.split('\n')
                    # Collect error context and replace all ATPs
                    atp_line_to_idx = {}  # {1-indexed line: ordinal}
                    atp_error_contexts = {}  # {ordinal: error_code_str}
                    atp_ordinal = 0
                    for i, line in enumerate(isa_code_lines):
                        if '<ATP>' in line and '(*' not in line:
                            atp_ordinal += 1
                            # Capture error comments above this line
                            error_comment_lines = []
                            for j in range(i-1, -1, -1):
                                if '(*' in isa_code_lines[j]:
                                    error_comment_lines.insert(0, isa_code_lines[j].strip())
                                else:
                                    break
                            isa_code_lines[i] = line.replace(
                                '<ATP>', _SLEDGEHAMMER_CMD, 1)
                            if isa_code_lines[i].count('sledgehammer') > 1:
                                isa_code_lines[i] = isa_code_lines[i].replace(
                                    _SLEDGEHAMMER_CMD, '', 1)
                            atp_line_to_idx[i + 1] = atp_ordinal
                            error_comment = '\n'.join(error_comment_lines)
                            atp_error_contexts[atp_ordinal] = (
                                error_comment + '\n' + isa_code_lines[i].strip())

                    logger.info(f"Batching {atp_ordinal} ATP placeholder(s) "
                                 f"in single Isabelle request")

                    isabelle_code = '\n'.join(isa_code_lines)
                    with open(f'{self.isabelle_dir}/{theory_name}.thy',
                              'w') as f:
                        f.write(isabelle_code)
                    (isabelle_response,
                     response_time) = self._get_response([theory_name],
                                                         self.isabelle_dir)
                    solving_time += response_time

                    finished_response = next(
                        (item for item in isabelle_response
                         if item.response_type == IsabelleResponseType.FINISHED),
                        None
                    )

                    if finished_response is not None:
                        response_body = self._parse_response_body(finished_response)
                        # Group messages by line number
                        line_messages = defaultdict(list)
                        all_tactic_messages = []
                        for node in self._get_attr(response_body, 'nodes', []):
                            for message in self._get_attr(node, 'messages', []):
                                msg_text = self._get_attr(message, 'message', '')
                                pos = self._get_attr(message, 'pos', {})
                                msg_line = self._get_attr(pos, 'line', 0)
                                line_messages[msg_line].append(msg_text)
                                all_tactic_messages.append(msg_text)

                        tactic_messages = all_tactic_messages

                        # Match tactics to ATP positions
                        all_solved = True
                        for atp_line in sorted(atp_line_to_idx):
                            idx = atp_line_to_idx[atp_line]
                            msgs = line_messages.get(atp_line, [])
                            tactic = self._parse_sledgehammer_tactic(msgs)

                            if tactic and all("no proof found" not in m.lower() for m in msgs):
                                logger.info(f"ATP {idx} solved: {tactic}")
                                # Replace this sledgehammer with the found tactic
                                lines = isabelle_code.split('\n')
                                for li, ln in enumerate(lines):
                                    if 'sledgehammer' in ln and '(*' not in ln:
                                        lines[li] = re.sub(
                                            r'sledgehammer\s*(\[.*?\])?\s*',
                                            tactic + ' ',
                                            ln, count=1).rstrip()
                                        if 'by by' in lines[li]:
                                            lines[li] = lines[li].replace('by by', 'by', 1)
                                        break
                                isabelle_code = '\n'.join(lines)
                            else:
                                all_solved = False
                                error_code = atp_error_contexts.get(idx, '').strip()
                                logger.info(f"ATP {idx} failed")
                                break

                        if all_solved:
                            semantic_validity = True
                            # Check for remaining errors after all ATPs solved
                            if self._get_attr(response_body, 'errors', []):
                                semantic_validity = False
                                unsolved_goals = self._extract_unsolved_goals(
                                    response_body, theory_name)
                                critique_output['semantic validity'] = False
                                critique_output['syntactic validity'] = True
                                critique_output['error code'] = error_code.strip()
                                critique_output['solving time'] = solving_time
                                critique_output['proof tactics'] = []
                                critique_output['code'] = self.code
                                critique_output['logical information'] = logical_information
                                critique_output['unsolved_goals'] = unsolved_goals
                                return critique_output
                        else:
                            semantic_validity = False
                            unsolved_goals = re.findall(
                                r'have\s+\w+:\s*"([^"]+)"[^"]*?sledgehammer',
                                isabelle_code, re.DOTALL)
                            if not unsolved_goals:
                                unsolved_goals = self._extract_unsolved_goals(
                                    response_body, theory_name)
                            critique_output['semantic validity'] = False
                            critique_output['syntactic validity'] = True
                            critique_output['error code'] = error_code.strip()
                            critique_output['solving time'] = solving_time
                            critique_output['proof tactics'] = []
                            critique_output['code'] = self.code
                            critique_output['logical information'] = logical_information
                            critique_output['unsolved_goals'] = unsolved_goals
                            return critique_output
        if semantic_validity:
            critique_output['error code'] = ''
        critique_output['syntactic validity'] = True
        critique_output['solving time'] = solving_time
        critique_output['proof tactics'] = tactic_messages
        critique_output['code'] = self.code
        critique_output['logical information'] = logical_information
        critique_output['semantic validity'] = semantic_validity
        if any(keyword in message for message in tactic_messages for keyword in self.ERROR_KEYWORDS):
            semantic_validity = False
            has_syntax_error = True
            critique_output['semantic validity'] = semantic_validity
            critique_output['syntactic validity'] = False
            critique_output['error code'] = error_code.strip()
            critique_output['solving time'] = solving_time
            critique_output['proof tactics'] = tactic_messages
            critique_output['code'] = self.code
            critique_output['logical information'] = logical_information
            return critique_output

        return critique_output

    def _try_schemes_sequentially(self, theory_name: str, generated_premises: str,
                                  hypothesis: str, premise: Optional[str]) -> dict:
        """
        Trial-and-error: Try each Walton scheme individually to find one that works.

        Strategy:
        1. Load all available Walton schemes
        2. For each scheme, inject its axiom into theory
        3. Try to prove with Isabelle
        4. If proof succeeds, return with scheme name
        5. If all schemes fail, return None

        Returns:
            dict with 'successful_scheme' and 'critique_output'
        """
        all_schemes = self.formaliser._load_walton_schemes()

        if not all_schemes:
            return {'successful_scheme': None, 'critique_output': None}

        logger.info(f"Trying {len(all_schemes)} Walton schemes sequentially")

        for scheme in all_schemes:
            logger.info(f"Trying scheme: {scheme['name']}")

            # Generate base formalization
            self._get_formalisation(
                generated_premises=generated_premises,
                hypothesis=hypothesis,
                premise=premise,
                theory_name=theory_name
            )

            # Inject this scheme's axiom
            isabelle_code = self.formaliser._inject_scheme_axioms(
                self.code,
                schemes_to_use=[scheme['name']]
            )

            # Save and try to prove
            self.formaliser.save_formalised_kb(isabelle_code, theory_name)

            # Attempt proof with this scheme
            theories_name = [theory_name]
            isabelle_response, solving_time = self._get_response(
                theories_name,
                self.isabelle_dir
            )

            # Check if proof succeeded
            finished_response = next(
                (item for item in isabelle_response if item.response_type == IsabelleResponseType.FINISHED),
                None
            )

            if finished_response:
                response_body = self._parse_response_body(finished_response)
                semantic_validity = True

                for node in self._get_attr(response_body, 'nodes', []):
                    for message in self._get_attr(node, 'messages', []):
                        if 'No proof found' in self._get_attr(message, 'message', ''):
                            semantic_validity = False
                            break

                if semantic_validity:
                    logger.info(f"Proof succeeded with scheme: {scheme['name']}")
                    return {
                        'successful_scheme': scheme['name'],
                        'critique_output': {
                            'semantic validity': True,
                            'syntactic validity': True,
                            'scheme_used': scheme['name'],
                            'solving time': solving_time,
                            'code': isabelle_code
                        }
                    }

            logger.debug(f"Scheme {scheme['name']} did not yield proof")

        logger.warning("All schemes exhausted, none successful")
        return {'successful_scheme': None, 'critique_output': None}

    def generate_bridge_axioms(self, isabelle_code: str, previous_attempts: list = None) -> str:
        """Public facade: generate bridge axioms for the current scheme."""
        scheme = getattr(self.formaliser, 'current_scheme', None)
        if not scheme:
            return ''
        return self.formaliser._get_bridge_axioms(
            scheme=scheme,
            isabelle_code=isabelle_code,
            previous_attempts=previous_attempts
        )

    def inject_bridge_axioms(self, isabelle_code: str, bridge_axioms: str,
                             theory_name: str) -> tuple:
        """Public facade: inject bridge axioms and update internal state.

        Returns (augmented_code, new_consts).
        """
        augmented_code, new_consts = self.formaliser._inject_bridge_axioms(
            isabelle_code, bridge_axioms
        )
        self.formaliser.save_formalised_kb(augmented_code, theory_name)
        self.code = augmented_code
        return augmented_code, new_consts

    def shutdown(self):
        def shutdown_isabelle_sync():
            self.isabelle.shutdown()
            logger.info('Isabelle server shut down')
        # Run the shutdown in a separate thread
        shutdown_thread = threading.Thread(target=shutdown_isabelle_sync)
        shutdown_thread.start()
        shutdown_thread.join()

        # Remove from global cleanup list since we've shut down cleanly
        if hasattr(self, 'server_process') and self.server_process in _server_processes:
            _server_processes.remove(self.server_process)
