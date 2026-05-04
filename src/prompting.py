from src.models import FunctionDefinition


def function_definition_for_prompt(functions: list[FunctionDefinition]) -> str:
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
    return (
        "You are a function selector.\n"
        "Available functions:\n"
        f"{function_definition_for_prompt(functions)}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Generate only this JSON object, without spaces or markdown:\n"
        '{"name":"<function_name>","parameters":{...}}\n'
    )
