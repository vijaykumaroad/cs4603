"""
chat/registry.py
Prompt registry management tool and interactive builder.

Usage as a module:
    from registry import PromptRegistry
    registry = PromptRegistry("prompts")
    prompt_str = registry.render("my_prompt", topic="AI")

Usage as a CLI:
    python chat/registry.py          # Interactive menu
    python chat/registry.py list     # List existing prompts
    python chat/registry.py create   # Create a new prompt
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
from jinja2 import Environment, Template, UndefinedError, meta

class PromptRegistry:
    """
    Manages loading, rendering, building, and validating YAML prompt templates.
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        """Initializes the registry and loads all YAML files from the directory."""
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_templates()
    
    # ------------------------------------------------------------------
    # Core Operations (Read & Render)
    # ------------------------------------------------------------------
    
    def _load_templates(self):
        """Scans the prompts directory and loads all YAML files into the registry."""
        self.templates.clear()
        yaml_files = list(self.prompts_dir.glob("*.yaml")) + list(self.prompts_dir.glob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and 'name' in data:
                        self.templates[data['name']] = data
            except Exception as e:
                print(f"[ERROR] Loading {yaml_file.name}: {e}")
    
    def get(self, name: str) -> Dict[str, Any]:
        """Retrieves a prompt template dict by name."""
        if name not in self.templates:
            raise KeyError(f"Prompt '{name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[name]
    
    def render(self, name: str, **kwargs) -> str:
        """Renders a prompt template with Jinja2 variable substitution."""
        prompt_data = self.get(name)
        template_str = prompt_data.get('template', '')
        
        try:
            template = Template(template_str)
            return template.render(**kwargs)
        except UndefinedError as e:
            raise ValueError(f"Missing template variable for '{name}': {e}")
            
    def list_prompts(self) -> List[str]:
        """Returns a list of all loaded prompt names."""
        return list(self.templates.keys())
    
    # ------------------------------------------------------------------
    # CRUD & Validation (Programmatic)
    # ------------------------------------------------------------------

    def validate_prompt(self, name: str) -> Tuple[bool, List[str], str]:
        """
        Validates the syntax and extracts expected variables for a prompt.
        Returns: (is_valid, list_of_variables, error_message)
        """
        if name not in self.templates:
            return False, [], f"Prompt '{name}' not found."
            
        template_str = self.templates[name].get('template', '')
        env = Environment()
        
        try:
            ast = env.parse(template_str)
            variables = list(meta.find_undeclared_variables(ast))
            return True, variables, ""
        except Exception as e:
            return False, [], str(e)

    def create(self, name: str, template: str, description: str = "", model: str = "") -> Path:
        """Programmatically creates and saves a new prompt template."""
        if name in self.templates:
            raise ValueError(f"Prompt '{name}' already exists.")
            
        filepath = self.prompts_dir / f"{name}.yaml"
        data = {
            "name": name,
            "version": "1.0",
            "model": model,
            "description": description,
            "template": template.strip()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        self.templates[name] = data
        return filepath

    def update(self, name: str, template: Optional[str] = None, description: Optional[str] = None) -> Path:
        """Updates an existing prompt template."""
        if name not in self.templates:
            raise KeyError(f"Prompt '{name}' not found.")
            
        data = self.templates[name]
        if template is not None:
            data['template'] = template.strip()
        if description is not None:
            data['description'] = description
            
        filepath = self.prompts_dir / f"{name}.yaml"
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        self._load_templates()
        return filepath

    def delete(self, name: str):
        """Deletes a prompt template from disk and memory."""
        if name not in self.templates:
            raise KeyError(f"Prompt '{name}' not found.")
            
        filepath = self.prompts_dir / f"{name}.yaml"
        if filepath.exists():
            filepath.unlink()
            
        del self.templates[name]

    # ------------------------------------------------------------------
    # Interactive CLI Builder Methods
    # ------------------------------------------------------------------

    def _ask(self, prompt: str, default: str = "") -> str:
        """Helper for standard CLI input."""
        suffix = f" [{default}]: " if default else ": "
        val = input(prompt + suffix).strip()
        return val if val else default

    def _ask_multiline(self, prompt: str) -> str:
        """Helper for multiline CLI input."""
        print(f"\n{prompt}")
        print("(Enter an empty line to finish)")
        lines = []
        while True:
            line = input("> ")
            if not line.strip() and lines:  # Stop on empty line if we have content
                break
            lines.append(line)
        return "\n".join(lines).strip()

    def _cli_create(self):
        """Interactive prompt creation workflow."""
        print("\n### CREATE NEW PROMPT ###")
        name = self._ask("Prompt name (lowercase, no spaces)")
        if not name or name in self.templates:
            print(f"❌ Invalid name or prompt '{name}' already exists.")
            return
            
        description = self._ask("Description (short summary)")
        model = self._ask("Target Model", "databricks-meta-llama-3-1-70b-instruct")
        
        print("\nHow do you want to build the template?")
        print("  1. PTCF-guided (Purpose → Task → Context → Format)")
        print("  2. Custom (write your own)")
        choice = self._ask("Choose", "1")
        
        if choice == "1":
            print("\nPTCF = Purpose + Task + Context + Format")
            purpose = self._ask_multiline("1. PURPOSE: What goal are we solving?")
            task = self._ask_multiline("2. TASK: What exactly should the model do?")
            context = self._ask_multiline("3. CONTEXT: What background rules apply?")
            format_req = self._ask_multiline("4. FORMAT: How should output be structured?")
            
            template = (
                f"**Purpose:** {purpose}\n\n"
                f"**Task:** {task}\n\n"
                f"**Context:** {context}\n\n"
                f"**Format:** {format_req}\n\n"
                f"**Input:** {{{{ user_input }}}}"
            )
        else:
            print("\nWrite your prompt template. Use {{ variable_name }} for substitutions.")
            template = self._ask_multiline("Enter your template:")

        print(f"\nReview Template:\n{'-'*40}\n{template}\n{'-'*40}")
        if self._ask("Save this prompt? (y/n)", "y").lower().startswith('y'):
            path = self.create(name, template, description, model)
            print(f"✓ Saved to {path}")

    def _cli_list(self):
        """Interactive prompt listing."""
        if not self.templates:
            print("\nNo prompts found.")
            return
            
        print("\n=== AVAILABLE PROMPTS ===")
        for i, (name, data) in enumerate(self.templates.items(), 1):
            desc = data.get('description', 'No description')
            print(f"{i}. {name}\n   └─ {desc}")

    def _cli_validate(self):
        """Interactive prompt validation."""
        name = self._ask("\nEnter prompt name to validate")
        is_valid, vars, err = self.validate_prompt(name)
        
        if err and not is_valid:
            print(f"❌ Error: {err}")
        else:
            print(f"✓ Template syntax is valid.")
            print(f"✓ Extracted Variables to supply in code: {vars if vars else 'None'}")

    def _cli_delete(self):
        """Interactive prompt deletion."""
        name = self._ask("\nEnter prompt name to delete")
        if name in self.templates:
            if self._ask(f"Are you sure you want to delete '{name}'? (y/n)", "n").lower().startswith('y'):
                self.delete(name)
                print(f"✓ Deleted '{name}'")
        else:
            print(f"❌ Prompt '{name}' not found.")

    def run_cli(self):
        """Main interactive menu loop."""
        while True:
            print("\n=== PROMPT REGISTRY & BUILDER ===")
            print("1. List prompts")
            print("2. Create new prompt")
            print("3. Validate prompt")
            print("4. Delete prompt")
            print("0. Exit")
            
            choice = self._ask("\nChoose", "0")
            if choice == "1": self._cli_list()
            elif choice == "2": self._cli_create()
            elif choice == "3": self._cli_validate()
            elif choice == "4": self._cli_delete()
            elif choice == "0": break


if __name__ == "__main__":
    registry = PromptRegistry()
    
    # Simple argument routing
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "list": registry._cli_list()
        elif cmd == "create": registry._cli_create()
        elif cmd == "validate" and len(sys.argv) > 2: 
            is_val, v, e = registry.validate_prompt(sys.argv[2])
            print(f"Valid: {is_val} | Vars: {v} | Error: {e}")
        else:
            print("Unknown command. Run without args for interactive mode.")
    else:
        registry.run_cli()