from json import loads
from os.path import exists
from re import search
from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message


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


@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False)
class JSONEntityExtractor(GraphComponent, EntityExtractorMixin):

    def __init__(self, parameters: Dict[Text, Any]):
        super(JSONEntityExtractor, self).__init__()

        if "path" in parameters.keys():
            self._path = parameters["path"]
        else:
            self._path = "./entities.json"

        if exists(self._path):
            with open(self._path, encoding="utf-8-sig") as ef:
                self.entity_model = self._load_entities_from_json(loads(ef.read()))
        else:
            self.entity_model = EntityModel([])

    @classmethod
    def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> GraphComponent:
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:
        # print(f"Process JSON Entities with ... {abspath(self._path)}")
        for message in messages:
            content = message.get("text")
            entities = self._recognize_entities(content)
            message.set("entities", message.get("entities", []) + entities, add_to_output=True)
        return messages

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
