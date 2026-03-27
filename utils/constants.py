import re


SAMPLE_SIZE_OUT = 5
SAMPLE_SIZE_IN = 5

THRESHOLDS = (4, 3, 2, 1)

NOT_COMMONS = ['base.', 'common.']

EXCLUDED_PREDICATE_PREFIXES = ["freebase_DOT_", "dataworld_DOT_", "schema_DOT_", "rdf_DOT_", "rdfs_DOT_", "owl_DOT_",
                               "xsd_DOT_", "user_DOT_", "ontology_DOT_", "type_DOT_object_DOT_key",
                               "common_DOT_topic_DOT_webpage", "common_DOT_image_DOT_appears_in_topic_gallery",
                               "common_DOT_webpage_DOT_topic", "common_DOT_topic_DOT_official_website", "base_DOT_",
                               "community_DOT_", "base_DOT_fbontology", "base_DOT_ontologies", "base_DOT_skosbase",
                               "base_DOT_aareas", "base_DOT_natlang", "type_DOT_object_DOT_permission",
                               "type_DOT_permission_DOT_controls", "common_DOT_topic_DOT_article"]

META_TYPE_PREFIXES = ("type", "common", "freebase", "schema", "rdf", "rdfs", "owl", "xsd", "user", "ontology", "base")
META_PRED_PREFIXES = (
    "type.object.",
    "type.type.",
)
# --- REGEX (defined globally for multiprocessing) ---
LITERAL_PREDICATES = {"type.object.name", "common.topic.alias"}
TYPE_PREDICATE = "type.object.type"

LITERAL_PREDICATES_SAN = {"type_DOT_object_DOT_name", "common_DOT_topic_DOT_alias",
                          "kg_DOT_object_profile_DOT_prominent_type"}
TYPE_PREDICATE_SAN = "type_DOT_object_DOT_type"

# anchored and predicate-filtered pattern for literals
LITERAL_PATTERN = re.compile(
    r'^<http://rdf.freebase.com/ns/([mg]\.[^>]*)>\s+'
    r'<http://rdf.freebase.com/ns/(type\.object\.name|common\.topic\.alias)>\s+'
    r'"(.*?)"(?:@([a-zA-Z\-]+))?\s*\.$'
)

# simpler, anchored-ish pattern for triples with object as URI
TYPE_PATTERN = re.compile(
    r'^<http://rdf.freebase.com/ns/([mg]\.[^>]*)>\s+'
    r'<http://rdf.freebase.com/ns/([^>]*)>\s+'
    r'<http://rdf.freebase.com/ns/([^>]*)>\s*\.$'
)