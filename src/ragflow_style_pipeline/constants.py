"""Single, immutable contract for 12345 entity extraction v1."""

PIPELINE_VERSION = "entity_extraction_v1"
DOCUMENT_SCHEMA_VERSION = "entity_document_v1"
ENTITY_SCHEMA_VERSION = "grounded_issue_entities_v1"
REJECT_SCHEMA_VERSION = "entity_extraction_reject_v1"
LINK_SCHEMA_VERSION = "entity_issue_links_v1"
PII_REDACTION_VERSION = "pii_redaction_v1"
PROMPT_SCHEMA_VERSION = "qwen_issue_string_arrays_v1"
GROUNDING_VERSION = "unicode_exact_substring_v1"
PROJECTION_VERSION = "issue_hyperedge_projection_v1"

DOC_ID_NAMESPACE = "12345-work-order/entity-extraction-v1"

CLEAN_FIELDS = (
    "title_clean",
    "case_content_clean",
    "case_goal_clean",
    "address_detail_clean",
)
CLEAN_FIELD_SOURCES = {
    "title_clean": "title",
    "case_content_clean": "case_content",
    "case_goal_clean": "case_goal",
    "address_detail_clean": "address_detail",
}
CLEAN_FIELD_LABELS = {
    "title_clean": "标题",
    "case_content_clean": "诉求内容",
    "case_goal_clean": "诉求目标",
    "address_detail_clean": "地址详情",
}

ENTITY_ROLES = ("objects", "problems", "questions", "locations", "requests")

SOURCE_ID_COLUMNS = ("id", "order_id")
METADATA_SOURCES = {
    "service_object_type": "service_object_type",
    "area_code_city": "area_code_city",
    "area_code_area": "area_code_area",
    "area_code_street": "area_code_street",
    "type1": "case_accord_type_one_name",
    "type2": "case_accord_type_two_name",
    "type3": "case_accord_type_three_name",
    "order_source": "order_source",
    "order_type": "order_type",
    "order_status": "order_status",
    "call_time": "call_time",
}
METADATA_LABELS = {
    "service_object_type": "诉求类型",
    "area_code_city": "市级区域",
    "area_code_area": "区县区域",
    "area_code_street": "街道区域",
    "type1": "一级业务分类",
    "type2": "二级业务分类",
    "type3": "三级业务分类",
    "order_source": "来源渠道",
    "order_type": "工单类型",
    "order_status": "工单状态",
    "call_time": "来电时间",
}

REQUIRED_TSV_COLUMNS = tuple(
    dict.fromkeys((*SOURCE_ID_COLUMNS, *CLEAN_FIELD_SOURCES.values(), *METADATA_SOURCES.values()))
)

DOCUMENT_PRIVATE_NAME = "documents.private.jsonl"
ENTITIES_PRIVATE_NAME = "entities.private.jsonl"
REJECTS_PRIVATE_NAME = "rejects.private.jsonl"
LINKS_PRIVATE_NAME = "entity_links.private.jsonl"
CONTRACT_PRIVATE_NAME = "run.contract.private.json"
CHECKPOINT_PRIVATE_NAME = "extraction.checkpoint.private.jsonl"
LOCK_PRIVATE_NAME = "extraction.lock.private"
PREPARE_SAFE_NAME = "prepare.safe.json"
DIAGNOSTICS_SAFE_NAME = "diagnostics.safe.jsonl"
RUN_SAFE_NAME = "run.safe.json"

SAFE_FORBIDDEN_KEYS = frozenset(
    {
        "doc_id",
        "source_id",
        "source_order_id",
        "text",
        "rag_text",
        "surface",
        "evidence",
        "prompt",
        "raw_response",
        "response",
        "content",
        "clean_fields",
        "mentions",
        "path",
        "model_path",
        "input_path",
    }
)
