from pydantic import BaseModel, Field

class Option(BaseModel):
    name: str
    desc: str # A concise description for the option, making embedding-based retrieval more effective.

class Aspect(BaseModel):
    name: str
    desc: str # A concise description for the aspect, making embedding-based retrieval more effective.
    options: list[Option] = Field(default_factory=list)

class Taxonomy(BaseModel):
    aspects: list[Aspect] = Field(default_factory=list)

# Example Structure 
_data = {
    "aspects": [
        {
            "name": "Color",
            "desc": "Primary hue of the object",
            "options": [{"name": "Red", "desc": "Bright crimson"}]
        }
    ]
}