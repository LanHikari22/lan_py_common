from classes import typeclass

@typeclass
def to_json(instance) -> str:
    """
    Converts instance to JSON
    """

@typeclass
def describe(instance) -> str:
    """
    Offers a description for the object
    """