from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, overload

from pydantic import BaseModel, TypeAdapter, ValidationError
from ruamel.yaml import YAML


@dataclass
class YamlLoc:
    line: int
    column: int
    file: Optional[Path] = None

    def __str__(self):
        file_str = ""
        if self.file is not None:
            file_str = (str(self.file) if self.file.is_absolute() else "./" + str(self.file)) + ":"
        return f"{file_str}{self.line}:{self.column}"

    def rich_str(self):
        if self.file is not None:
            file = self.file.absolute()
            return f"[link=file://{str(file)}:{self.line}:{self.column}]{self.file}:{self.line}:{self.column}[/link]"
        else:
            return f"{self.line}:{self.column}"

    @classmethod
    def from_loc(cls, loc: tuple[int | str, ...], yaml_data, line_offset: int = 0, file: Optional[Path] = None):
        item: Any = yaml_data
        loc_keys = []
        loc_items = [item]
        for item_loc in loc:
            if isinstance(item, str) and item.startswith("$"):
                # Variable references don't have a location in the YAML file, so we skip them.
                break
            try:
                next_item = item[item_loc]  # type: ignore
            except Exception:
                pass
            else:
                loc_keys.append(item_loc)
                loc_items.append(next_item)
                item = next_item

        if len(loc_items) >= 2:
            line, col = loc_items[-2].lc.value(loc_keys[-1])
        else:
            lc = loc_items[-1].lc
            if isinstance(lc, tuple):
                line, col = lc
            else:
                line, col = lc.line, lc.col
        return cls(line=line + 1 + line_offset, column=col + 1, file=file), tuple(loc_keys)


class YamlDocument:
    def __init__(self, yaml_str: str, line_offset: int = 0, file: Optional[Path] = None):
        self.yaml_str = yaml_str
        self.line_offset = line_offset
        self.file = file

    @overload
    @classmethod
    def read_file(cls, file: str | Path, doc_id: None = None) -> list[YamlDocumentWithFile]: ...
    @overload
    @classmethod
    def read_file(cls, file: str | Path, doc_id: int) -> YamlDocumentWithFile: ...
    @classmethod
    def read_file(
        cls, file: str | Path, doc_id: int | None = None
    ) -> YamlDocumentWithFile | list[YamlDocumentWithFile]:
        file = Path(file)
        with file.open("r") as f:
            yaml_str = f.read()

        # Split the YAML string into documents
        yaml_lines = yaml_str.splitlines(True)
        sep_lines = [-1] + [i for i, line in enumerate(yaml_lines) if line.strip() == "---"] + [len(yaml_lines)]
        n_doc = len(sep_lines) - 1

        if doc_id is None:
            return [
                YamlDocumentWithFile("".join(yaml_lines[sep_lines[i] + 1 : sep_lines[i + 1]]), sep_lines[i] + 1, file)
                for i in range(n_doc)
            ]
        elif doc_id >= n_doc:
            raise FileNotFoundError(f"Document {doc_id} not found in {file.name} (file contains {n_doc} documents).")
        else:
            l0, l1 = sep_lines[doc_id] + 1, sep_lines[doc_id + 1]
            yaml_str = "".join(yaml_lines[l0:l1])
            return YamlDocumentWithFile(yaml_str, l0, file)

    def validate[T](self, model: type[T], strict: Optional[bool] = None) -> T:
        yaml = YAML(typ="rt")
        data = yaml.load(self.yaml_str)

        try:
            if issubclass(model, BaseModel):
                return model.model_validate(data, strict=strict)
            else:
                adapter = TypeAdapter(model)
                return adapter.validate_python(data, strict=strict)
        except ValidationError as e:
            errors = e.errors()
            for error in errors:
                ctx = error.setdefault("ctx", {})
                ctx["yaml_loc"], ctx["loc"] = YamlLoc.from_loc(error["loc"], data, self.line_offset, self.file)
            raise ValidationError.from_exception_data(title=e.title, line_errors=errors) from None  # type: ignore


class YamlDocumentWithFile(YamlDocument):
    line_offset: int
    file: Path

    def __init__(self, yaml_str: str, line_offset, file: Path):
        super().__init__(yaml_str, line_offset, file)


class InvalidDocumentCountError(ValueError):
    def __init__(self, expected: int, actual: int, file: Optional[Path | str] = None):
        self.expected = expected
        self.actual = actual
        self.file = file
        file_str = f" in file {file}" if file is not None else ""
        super().__init__(f"Expected {expected} YAML document(s) but found {actual}{file_str}.")


def model_validate_yaml(yaml_str: str, model: type, strict: Optional[bool] = None):
    return YamlDocument(yaml_str).validate(model, strict=strict)


def model_validate_yaml_file[T](
    file: str | Path, model: type[T] | type, doc_id: int = 0, strict: Optional[bool] = None
) -> T:
    return YamlDocument.read_file(file, doc_id=doc_id).validate(model, strict=strict)


def pretty_validation_error_msg(e: ValidationError, model: type):
    msg = f"{len(e.errors())} validation error(s) found when parsing {model.__name__}:\n"
    for error in e.errors():
        if "ctx" in error and "yaml_loc" in error["ctx"]:
            item = "[bright_black]"
            loc = error["ctx"]["loc"]
            for i, k in enumerate(loc):
                v = str(k)
                if i == len(loc) - 1:
                    item += "[/bright_black]"
                    v = f"[b]{v}[/b]"
                if isinstance(k, int):
                    item += f"[{v}]"
                elif i > 0:
                    item += f".{v}"
                else:
                    item += v
            loc = error["ctx"]["yaml_loc"].rich_str()
            msg += f"{item} [i]at {loc}[/i]\n"
        input_str = repr(error.get("input", ""))
        msg_line = f"  {error['msg']} [bright_black](input value: {input_str}"
        if len(msg_line) > 120:
            input_str = input_str[:120] + "..."
        msg += msg_line + ")[/bright_black]\n"
    return msg
