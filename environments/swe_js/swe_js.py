"""
Minimal SWE-JS environment for multi-turn code repair with RL.

Uses SWE-bench/SWE-smith-js dataset with prime-sandboxes.
Agent gets bash access to fix bugs in JS repos.
"""

import json
import re
import verifiers as vf
from datasets import load_dataset, Dataset


SYSTEM_PROMPT = """You are a software engineer fixing a bug in a JavaScript repository.

You have access to:
- bash: Run shell commands (ls, cat, grep, sed, etc.)
- submit: Call this when you have fixed the bug

The repository has been cloned to /workspace with dependencies installed.

Workflow:
1. Read the bug report carefully
2. Explore the codebase to find relevant code
3. Make minimal changes to fix the bug
4. Test your fix with npm test
5. Call submit() when done
"""


def load_environment(
    num_examples: int = -1,
    max_turns: int = 100,
    timeout_per_command: int = 60,
    test_timeout: int = 300,
    **kwargs,
) -> vf.Environment:
    """
    Load the SWE-JS environment.

    Args:
        num_examples: Number of examples to load (-1 for all)
        max_turns: Maximum conversation turns
        timeout_per_command: Timeout for each bash command in seconds
        test_timeout: Timeout for running tests in seconds
    """
    # Load dataset
    ds = load_dataset("SWE-bench/SWE-smith-js", split="train")
    if num_examples > 0:
        ds = ds.select(range(min(num_examples, len(ds))))

    # Transform to verifiers format
    examples = []
    for row in ds:
        # Parse repo info
        repo = row["repo"]  # e.g., "swesmith/Automattic__mongoose.5f57a5bb"
        instance_id = row["instance_id"]

        # Build prompt
        prompt_content = f"""# Bug Report

{row["problem_statement"]}

---

The repository is at /workspace. Fix the bug, test it, then call submit()."""

        examples.append({
            "prompt": [{"role": "user", "content": prompt_content}],
            "answer": row["patch"],  # Bug-introducing patch (not the fix)
            "info": json.dumps({
                "instance_id": instance_id,
                "repo": repo,
                "fail_to_pass": row["FAIL_TO_PASS"],
                "pass_to_pass": row["PASS_TO_PASS"],
                "test_timeout": test_timeout,
            }),
        })

    dataset = Dataset.from_list(examples)

    # Rubric: always run tests at end and compute pass rate
    async def test_pass_rate(completion, info, state) -> float:
        """Run FAIL_TO_PASS tests and return fraction that pass."""
        info_data = json.loads(info) if isinstance(info, str) else info
        fail_to_pass = info_data.get("fail_to_pass", [])

        if not fail_to_pass:
            return 0.0

        # Tests are always run in post_rollout, results stored in state
        test_results = state.get("test_results", {})
        if not test_results:
            return 0.0

        passed = sum(1 for t in fail_to_pass if test_results.get(t, False))
        return passed / len(fail_to_pass)

    rubric = vf.Rubric(funcs=[test_pass_rate])

    # Create the environment
    env = SweJsEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
        timeout_per_command=timeout_per_command,
    )

    return env


class SweJsEnv(vf.SandboxEnv):
    """Multi-turn environment for JS bug fixing using tool calling."""

    def __init__(
        self,
        timeout_per_command: int = 60,
        **kwargs,
    ):
        super().__init__(
            docker_image="node:20",
            timeout_per_command_seconds=timeout_per_command,
            memory_gb=4,
            cpu_cores=2,
            **kwargs,
        )
        # Add submit tool
        self.add_tool(self.submit)

    async def submit(self) -> str:
        """Submit your fix. Call this when you have fixed the bug and tested it."""
        return "Submission received. Evaluating tests..."

    async def _run_bash(self, state: vf.State, command: str) -> str:
        """Helper to run bash command with proper state params."""
        return await self.bash(
            command,
            sandbox_id=state["sandbox_id"],
            sandbox_state=state["sandbox_state"],
        )

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Clone repo, checkout buggy branch, install deps."""
        state = await super().setup_state(state, **kwargs)

        info = json.loads(state["info"]) if isinstance(state["info"], str) else state["info"]
        repo = info["repo"]
        instance_id = info["instance_id"]

        # Install git (node:20 already has most deps including libcurl)
        await self._run_bash(state, "apt-get update && apt-get install -y git")

        clone_url = f"https://github.com/{repo}.git"
        await self._run_bash(state, f"git clone {clone_url} /workspace")
        await self._run_bash(state, f"cd /workspace && git checkout {instance_id}")
        await self._run_bash(state, "cd /workspace && npm install")

        # Set working directory for future commands
        state["working_dir"] = "/workspace"

        return state

    @staticmethod
    def _parse_mocha_log(log: str) -> dict[str, str]:
        """Parse mocha test output into per-test pass/fail map.

        Adapted from SWE-smith's parse_log_mocha.
        """
        test_status_map = {}
        # Match ✓/✔ (pass), ✖ (fail), - (skip)
        pattern = r"^\s*([✓✔]|✖|-)\s(.+?)(?:\s\((\d+\s*m?s)\))?$"
        # Match numbered failures like "1) test name"
        fail_pattern = r"^\s*\d+\)\s+(.+?)(?:\s\((\d+\s*m?s)\))?$"

        for line in log.split("\n"):
            match = re.match(pattern, line.strip())
            if match:
                status_symbol, test_name, _ = match.groups()
                if status_symbol in ("✓", "✔"):
                    test_status_map[test_name.strip()] = "PASSED"
                elif status_symbol == "✖":
                    test_status_map[test_name.strip()] = "FAILED"
            else:
                fail_match = re.match(fail_pattern, line.strip())
                if fail_match:
                    test_status_map[fail_match.group(1).strip()] = "FAILED"

        return test_status_map

    async def post_rollout(self, state: vf.State, **kwargs) -> vf.State:
        """Always run tests after rollout ends, regardless of how it stopped."""
        info = json.loads(state["info"]) if isinstance(state["info"], str) else state["info"]
        fail_to_pass = info.get("fail_to_pass", [])

        test_results = {}

        try:
            # Run full test suite once, parse per-test results
            output = await self._run_bash(
                state,
                "cd /workspace && timeout 300 npm test 2>&1 || true"
            )
            parsed = self._parse_mocha_log(output)

            # Match FAIL_TO_PASS test names against parsed results
            for test_name in fail_to_pass:
                # Try exact match first
                if test_name in parsed:
                    test_results[test_name] = parsed[test_name] == "PASSED"
                else:
                    # Fuzzy: check if test_name is a substring of any parsed test
                    matched = False
                    for parsed_name, status in parsed.items():
                        if test_name in parsed_name or parsed_name in test_name:
                            test_results[test_name] = status == "PASSED"
                            matched = True
                            break
                    if not matched:
                        test_results[test_name] = False
        except Exception:
            for test_name in fail_to_pass:
                test_results[test_name] = False

        state["test_results"] = test_results
        return await super().post_rollout(state, **kwargs)
