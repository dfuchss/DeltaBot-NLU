from json import loads
from os.path import exists
from re import search
from typing import Dict, Optional, Text, Any, List

from rasa.nlu.extractors.extractor import EntityExtractor

_entities = "./entities.json"


class Entity:
    """Defines an entity."""

    def __init__(self, name: str, values: List[str]) -> None:
        """ Initialize Entity.

        :param name the name of the entity
        :param values the values / synonyms of the entity
        """
        self.name = name
        self.values = list(map(lambda s: s.lower(), values))


class EntityGroup:
    """ Defines a group of entities. """

    def __init__(self, name: str, entities: List[Entity] = None) -> None:
        """
        Initialize group of entities.

        :param name the name of the group
        :param entities the initial set of entities
        """
        if entities is None:
            entities = []
        self.entities = entities
        self.name = name

    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to group.

        :param entity the entity
        """
        self.entities.append(entity)


class EntityModel:
    """ Defines a serializable Entity Model. """

    def __init__(self, groups: List[EntityGroup]):
        """
        Create a new Entity Model.

        :param groups the groups in the model.
        """
        self.groups = groups


class JSONEntityExtractor(EntityExtractor):
    def __init__(self, parameters: Optional[Dict[Text, Any]]):
        super().__init__(parameters)

        if exists(_entities):
            with open(_entities, encoding="utf-8-sig") as ef:
                self.entity_model = self._load_entities_from_json(loads(ef.read()))
        else:
            self.entity_model = EntityModel([])

    def process(self, message, **kwargs: Any) -> None:
        content = message.get("text")
        entities = self._recognize_entities(content)
        message.set("entities", message.get("entities", []) + entities, add_to_output=True)

    def _recognize_entities(self, content: str) -> List[dict]:
        """
        Identify mentioned entities in a string.

        :param content: the string that shall be searched for entities
        :return: a list of entity results
        """
        result = []
        content = content.lower()

        for group in self.entity_model.groups:
            for entity in group.entities:
                for value in entity.values:
                    match = search(f"\\b{value}\\b", content)
                    if match is not None:
                        er = {
                            "start": match.start(),
                            "end": match.end(),
                            "value": entity.name,
                            "confidence": 1.0,
                            "entity": group.name,
                        }
                        result.append(er)
                        break

        return result

    @staticmethod
    def _load_entities_from_json(data: dict) -> EntityModel:
        """
        Load a entity model from a dictionary.

        :param data: the input dictionary
        :return: the entity model
        """
        groups = []
        for group in data["groups"]:
            group_name = group["name"]
            group_entities = []
            for entity in group["entities"]:
                entity_name = entity["name"]
                entity_vals = entity["values"]
                group_entities.append(Entity(entity_name, entity_vals))
            groups.append(EntityGroup(group_name, group_entities))
        return EntityModel(groups)
