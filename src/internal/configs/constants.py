"""Shared application constants."""

from __future__ import annotations

import os
import re
from enum import Enum


class MessageType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL_RESPONSE = "tool_call_response"
    USER_REMINDER = "user_reminder"


class ChatMessageSimpleType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    FILE_TEXT = "file_text"


class DocumentSource(str, Enum):
    INGESTION_API = "ingestion_api"
    SLACK = "slack"
    WEB = "web"
    GOOGLE_DRIVE = "google_drive"
    GMAIL = "gmail"
    GITHUB = "github"
    GITBOOK = "gitbook"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    GURU = "guru"
    BOOKSTACK = "bookstack"
    OUTLINE = "outline"
    CONFLUENCE = "confluence"
    JIRA = "jira"
    SLAB = "slab"
    PRODUCTBOARD = "productboard"
    FILE = "file"
    CODA = "coda"
    CANVAS = "canvas"
    NOTION = "notion"
    ZULIP = "zulip"
    LINEAR = "linear"
    HUBSPOT = "hubspot"
    DOCUMENT360 = "document360"
    GONG = "gong"
    GOOGLE_SITES = "google_sites"
    ZENDESK = "zendesk"
    LOOPIO = "loopio"
    DROPBOX = "dropbox"
    SHAREPOINT = "sharepoint"
    TEAMS = "teams"
    SALESFORCE = "salesforce"
    DISCOURSE = "discourse"
    AXERO = "axero"
    CLICKUP = "clickup"
    MEDIAWIKI = "mediawiki"
    WIKIPEDIA = "wikipedia"
    ASANA = "asana"
    S3 = "s3"
    R2 = "r2"
    GOOGLE_CLOUD_STORAGE = "google_cloud_storage"
    OCI_STORAGE = "oci_storage"
    XENFORO = "xenforo"
    NOT_APPLICABLE = "not_applicable"
    DISCORD = "discord"
    FRESHDESK = "freshdesk"
    FIREFLIES = "fireflies"
    EGNYTE = "egnyte"
    AIRTABLE = "airtable"
    HIGHSPOT = "highspot"
    DRUPAL_WIKI = "drupal_wiki"
    IMAP = "imap"
    TESTRAIL = "testrail"
    MOCK_CONNECTOR = "mock_connector"
    USER_FILE = "user_file"
    CRAFT_FILE = "craft_file"


class FederatedConnectorSource(str, Enum):
    FEDERATED_SLACK = "federated_slack"

    def to_non_federated_source(self) -> DocumentSource | None:
        if self == FederatedConnectorSource.FEDERATED_SLACK:
            return DocumentSource.SLACK
        return None


DocumentSourceRequiringTenantContext: list[DocumentSource] = [DocumentSource.FILE]

DocumentSourceDescription: dict[DocumentSource, str] = {
    DocumentSource.INGESTION_API: "Documents ingested via API",
    DocumentSource.SLACK: "Team messages and channel discussions",
    DocumentSource.WEB: "Indexed web pages",
    DocumentSource.GOOGLE_DRIVE: "Documents, spreadsheets, and presentations",
    DocumentSource.GMAIL: "Email conversations and threads",
    DocumentSource.GITHUB: "Pull requests, issues, and code reviews",
    DocumentSource.GITBOOK: "Documentation and knowledge base pages",
    DocumentSource.GITLAB: "Merge requests, issues, and code",
    DocumentSource.BITBUCKET: "Pull requests, issues, and code",
    DocumentSource.GURU: "Knowledge cards and collections",
    DocumentSource.BOOKSTACK: "Documentation and knowledge base pages",
    DocumentSource.OUTLINE: "Documentation and knowledge base pages",
    DocumentSource.CONFLUENCE: "Wiki pages, spaces, and documentation",
    DocumentSource.JIRA: "Issues, tickets, and project tracking",
    DocumentSource.SLAB: "Documentation and knowledge base pages",
    DocumentSource.PRODUCTBOARD: "Product management boards and insights",
    DocumentSource.FILE: "Uploaded files",
    DocumentSource.CANVAS: "Courses, pages, and assignments",
    DocumentSource.CODA: "Documents, tables, and team workspace pages",
    DocumentSource.NOTION: "Documentation, notes, and project pages",
    DocumentSource.ZULIP: "Chat messages and topic discussions",
    DocumentSource.LINEAR: "Engineering and product tickets",
    DocumentSource.HUBSPOT: "CRM data, contacts, and deals",
    DocumentSource.DOCUMENT360: "Knowledge base articles and documentation",
    DocumentSource.GONG: "Sales call recordings and transcripts",
    DocumentSource.GOOGLE_SITES: "Website pages and content",
    DocumentSource.ZENDESK: "Support tickets and help articles",
    DocumentSource.LOOPIO: "RFP responses and content library",
    DocumentSource.DROPBOX: "Cloud-stored files and folders",
    DocumentSource.SHAREPOINT: "Documents and team sites",
    DocumentSource.TEAMS: "Chat messages and channels",
    DocumentSource.SALESFORCE: "Sales data, accounts, and opportunities",
    DocumentSource.DISCOURSE: "Community forums and discussions",
    DocumentSource.AXERO: "Employee engagement and intranet content",
    DocumentSource.CLICKUP: "Tasks and project management",
    DocumentSource.MEDIAWIKI: "Wiki pages and articles",
    DocumentSource.WIKIPEDIA: "Encyclopedia articles",
    DocumentSource.ASANA: "Tasks and project management",
    DocumentSource.S3: "Cloud-stored files and objects",
    DocumentSource.R2: "Cloud-stored files and objects",
    DocumentSource.GOOGLE_CLOUD_STORAGE: "Cloud-stored files and objects",
    DocumentSource.OCI_STORAGE: "Cloud-stored files and objects",
    DocumentSource.XENFORO: "Forum threads and discussions",
    DocumentSource.DISCORD: "Chat messages and server discussions",
    DocumentSource.FRESHDESK: "Support tickets and customer queries",
    DocumentSource.FIREFLIES: "Meeting transcripts and recordings",
    DocumentSource.EGNYTE: "Cloud-stored files and documents",
    DocumentSource.AIRTABLE: "Structured data and records",
    DocumentSource.HIGHSPOT: "Sales enablement content and pitches",
    DocumentSource.DRUPAL_WIKI: "Knowledge base pages and content",
    DocumentSource.IMAP: "Email messages and threads",
    DocumentSource.TESTRAIL: "Test cases and QA management",
}


class NotificationType(str, Enum):
    REINDEX = "reindex"
    PERSONA_SHARED = "persona_shared"
    TRIAL_ENDS_TWO_DAYS = "two_day_trial_ending"
    RELEASE_NOTES = "release_notes"
    ASSISTANT_FILES_READY = "assistant_files_ready"
    FEATURE_ANNOUNCEMENT = "feature_announcement"
    CONNECTOR_REPEATED_ERRORS = "connector_repeated_errors"
    LICENSE_EXPIRY_WARNING = "license_expiry_warning"
    SCHEDULED_TASK_FAILED = "scheduled_task_failed"
    SCHEDULED_TASK_AWAITING_APPROVAL = "scheduled_task_awaiting_approval"
    SCHEDULED_TASK_PRE_APPROVED_ACTION = "scheduled_task_pre_approved_action"
    APPROVAL_REQUESTED = "approval_requested"


class BlobType(str, Enum):
    R2 = "r2"
    S3 = "s3"
    GOOGLE_CLOUD_STORAGE = "google_cloud_storage"
    OCI_STORAGE = "oci_storage"


class DocumentIndexType(str, Enum):
    COMBINED = "combined"
    SPLIT = "split"


class AuthType(str, Enum):
    BASIC = "basic"
    GOOGLE_OAUTH = "google_oauth"
    OIDC = "oidc"
    SAML = "saml"
    CLOUD = "cloud"


class QueryHistoryType(str, Enum):
    DISABLED = "disabled"
    ANONYMIZED = "anonymized"
    NORMAL = "normal"


class SessionType(str, Enum):
    CHAT = "Chat"
    SEARCH = "Search"
    SLACK = "Slack"


class QAFeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    MIXED = "mixed"


class SearchFeedbackType(str, Enum):
    ENDORSE = "endorse"
    REJECT = "reject"
    HIDE = "hide"
    UNHIDE = "unhide"


class TokenRateLimitScope(str, Enum):
    USER = "user"
    USER_GROUP = "user_group"
    GLOBAL = "global"


class FileStoreType(str, Enum):
    S3 = "s3"
    POSTGRES = "postgres"
    GCS = "gcs"


class FileType(str, Enum):
    CSV = "text/csv"


class FileOrigin(str, Enum):
    CHAT_UPLOAD = "chat_upload"
    CHAT_IMAGE_GEN = "chat_image_gen"
    CONNECTOR = "connector"
    CONNECTOR_FILE_UPLOAD = "connector_file_upload"
    CONNECTOR_METADATA = "connector_metadata"
    GENERATED_REPORT = "generated_report"
    INDEXING_CHECKPOINT = "indexing_checkpoint"
    INDEXING_STAGING = "indexing_staging"
    PLAINTEXT_CACHE = "plaintext_cache"
    OTHER = "other"
    QUERY_HISTORY_CSV = "query_history_csv"
    SANDBOX_SNAPSHOT = "sandbox_snapshot"
    SKILL_BUNDLE = "skill_bundle"
    USER_FILE = "user_file"
    GENERATED = "generated"


class MilestoneRecordType(str, Enum):
    TENANT_CREATED = "tenant_created"
    USER_SIGNED_UP = "user_signed_up"
    VISITED_ADMIN_PAGE = "visited_admin_page"
    CREATED_CONNECTOR = "created_connector"
    CONNECTOR_SUCCEEDED = "connector_succeeded"
    RAN_QUERY = "ran_query"
    USER_MESSAGE_SENT = "user_message_sent"
    MULTIPLE_ASSISTANTS = "multiple_assistants"
    CREATED_ASSISTANT = "created_assistant"
    CREATED_SEARCH_BOT = "created_search_bot"
    REQUESTED_CONNECTOR = "requested_connector"
    CHAT = "chat"
    SEARCH = "search"


# Persona IDs
DEFAULT_PERSONA_ID: int = 0
DEFAULT_CC_PAIR_ID: int = 1

# Public API tag
PUBLIC_API_TAGS: list[str | Enum] = ["public"]

# Cookies
FASTAPI_USERS_AUTH_COOKIE_NAME: str = (
    os.environ.get("AUTH_COOKIE_NAME") or "fastapiusersauth"
)
TENANT_ID_COOKIE_NAME = "agentic_search_tid"
ANONYMOUS_USER_COOKIE_NAME = "agentic_search_anonymous_user"

# Anonymous/no-auth user identifiers
ANONYMOUS_USER_INFO_ID = "__anonymous_user__"
NO_AUTH_PLACEHOLDER_USER_UUID = "00000000-0000-0000-0000-000000000001"
NO_AUTH_PLACEHOLDER_USER_EMAIL = "no-auth-placeholder@danswer.ai"
ANONYMOUS_USER_UUID = "00000000-0000-0000-0000-000000000002"
ANONYMOUS_USER_EMAIL = "anonymous@danswer.ai"

# Password validation
PASSWORD_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"

# Credential masking
MASK_CREDENTIAL_CHAR = "•"
MASK_CREDENTIAL_LONG_RE = re.compile(r"^.{4}\.{3}.{4}$")

# Metadata field name for ignoring chunks from QA
IGNORE_FOR_QA = "ignore_for_qa"

# Deprecated key for porting gen AI key from old system
GEN_AI_API_KEY_STORAGE_KEY = "genai_api_key"

# Document index and retrieval constants
PUBLIC_DOC_PAT = "PUBLIC"
ID_SEPARATOR = ":;:"
DEFAULT_BOOST = 0

RETURN_SEPARATOR = "\n\n"
SECTION_SEPARATOR = "\n\n---\n\n"
INDEX_SEPARATOR = "__"

SOURCE_TYPE = "source_type"

CANCEL_CHECK_INTERVAL = 20
DISPATCH_SEP_CHAR = "\n"
FORMAT_DOCS_SEPARATOR = "\n\n"
NUM_EXPLORATORY_DOCS = 15

# Key-value store keys
KV_REINDEX_KEY = "kv_reindex_key"
KV_UNSTRUCTURED_API_KEY = "unstructured_api_key"
KV_USER_STORE_KEY = "INVITED_USERS"
KV_PENDING_USERS_KEY = "PENDING_USERS"
KV_ANONYMOUS_USER_PREFERENCES_KEY = "anonymous_user_preferences"
KV_ANONYMOUS_USER_PERSONALIZATION_KEY = "anonymous_user_personalization"
KV_CRED_KEY = "credential_id_{}"
KV_GMAIL_CRED_KEY = "gmail_app_credential"
KV_GMAIL_SERVICE_ACCOUNT_KEY = "gmail_service_account_key"
KV_GOOGLE_DRIVE_CRED_KEY = "google_drive_app_credential"
KV_GOOGLE_DRIVE_SERVICE_ACCOUNT_KEY = "google_drive_service_account_key"
KV_GEN_AI_KEY_CHECK_TIME = "genai_api_key_last_check_time"
KV_SETTINGS_KEY = "agentic_search_settings"
KV_CUSTOMER_UUID_KEY = "customer_uuid"
KV_INSTANCE_DOMAIN_KEY = "instance_domain"
KV_ENTERPRISE_SETTINGS_KEY = "agentic_search_enterprise_settings"
KV_CUSTOM_ANALYTICS_SCRIPT_KEY = "__custom_analytics_script__"
KV_KG_CONFIG_KEY = "kg_config"


BLURB_SIZE = 250

# Version patterns for Docker image tags
STABLE_VERSION_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
DEV_VERSION_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)-beta\.(\d+)$")

TMP_DRALPHA_PERSONA_NAME = "KG Beta"
