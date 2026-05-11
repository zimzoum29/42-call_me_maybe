"""Helpers that build human-readable function descriptions and prompts.

The functions here are used to generate the textual prompt sent to the
LLM, describing available functions in a compact, unambiguous form.
"""

from src.models import FunctionDefinition


def function_definition_for_prompt(functions: list[FunctionDefinition]) -> str:
    """Render a compact description of available functions.

    Args:
        functions: List of `FunctionDefinition` instances.

    Returns:
        A multi-line string describing each function name, description,
        parameters and return type suitable for inclusion in a prompt.
    """
    lines: list[str] = []
    for function in functions:
        lines.append(f"- name: {function.name}")
        lines.append(f"  description: {function.description}")
        if function.parameters:
            params = ", ".join(
                f"{name}: {definition.type}"
                for name, definition in function.ordered_parameter_items()
            )
        else:
            params = "none"
        lines.append(f"  parameters: {params}")
        lines.append(f"  returns: {function.returns.type}")
    return "\n".join(lines)


def build_generation_prompt(
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Construct the full prompt string sent to the LLM.

    The returned prompt includes the functions description followed by
    the user's request and an explicit instruction to emit a compact
    JSON object describing the chosen function call.
    """
    return (
        "You are a function selector.\n"
        "Available functions:\n"
        f"{function_definition_for_prompt(functions)}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Generate only this JSON object, without spaces or markdown:\n"
        '{"name":"<function_name>","parameters":{...}}\n'
    )
