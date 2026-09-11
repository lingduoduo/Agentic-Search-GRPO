"""User-group management API router.

Provides full CRUD for groups stored in AgenticSearchStore, mirroring the
structure of the user_group API. repo-specific concepts
(curators, personas, connector-credential pairs, document sets) are absent
from this repo and are not exposed here.

All endpoints require admin access (user ID or email in
AppSettings.auth.super_users). Authenticated non-admin users can query their
own group memberships via GET /manage/user-groups.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response

from src.internal.auth import AuthenticatedUser
from src.internal.servers.users.api import resolve_active_user
from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.db.models import GroupRecord
from src.internal.servers.user_group.models import AddUsersToGroupRequest
from src.internal.servers.user_group.models import RemoveUsersFromGroupRequest
from src.internal.servers.user_group.models import UserGroupCreate
from src.internal.servers.user_group.models import UserGroupRename
from src.internal.servers.user_group.models import UserGroupUpdate
from src.internal.servers.user_group.models import UserGroupView
from src.internal.servers._auth import make_require_admin

logger = logging.getLogger(__name__)


def create_user_group_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter for user-group endpoints bound to *store*."""

    router = APIRouter(prefix="/manage", tags=["user-groups"])

    _require_admin = make_require_admin(app_settings)

    def _require_user(request: Request) -> AuthenticatedUser:
        user = resolve_active_user(request, store)
        if user is None or user.is_anonymous:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return user

    def _get_group_or_404(group_id: str) -> GroupRecord:
        group = store.get_group(group_id)
        if group is None:
            raise HTTPException(
                status_code=404, detail=f"Group {group_id!r} not found."
            )
        return group

    @router.get("/admin/user-group")
    def list_user_groups(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[UserGroupView]:
        """Return all groups."""
        return [UserGroupView.from_record(g) for g in store.list_groups()]

    @router.get("/admin/user-group/{group_id}")
    def get_user_group(
        group_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> UserGroupView:
        """Return a single group by ID."""
        return UserGroupView.from_record(_get_group_or_404(group_id))

    @router.post("/admin/user-group", status_code=201)
    def create_user_group(
        body: UserGroupCreate,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> UserGroupView:
        """Create a new group. *group_id* is auto-generated if omitted."""
        group_id = body.group_id or f"group_{uuid4().hex}"
        if store.get_group(group_id) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Group {group_id!r} already exists.",
            )
        record = store.upsert_group(
            GroupRecord(id=group_id, name=body.name, user_ids=list(body.user_ids))
        )
        logger.info("Created group %s (%s).", group_id, body.name)
        return UserGroupView.from_record(record)

    @router.patch("/admin/user-group/{group_id}")
    def update_user_group(
        group_id: str,
        body: UserGroupUpdate,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> UserGroupView:
        """Update group name and/or full member list."""
        existing = _get_group_or_404(group_id)
        new_name = body.name if body.name is not None else existing.name
        new_user_ids = (
            body.user_ids if body.user_ids is not None else list(existing.user_ids)
        )
        record = store.upsert_group(
            GroupRecord(id=group_id, name=new_name, user_ids=new_user_ids)
        )
        return UserGroupView.from_record(record)

    @router.patch("/admin/user-group/{group_id}/rename")
    def rename_user_group(
        group_id: str,
        body: UserGroupRename,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> UserGroupView:
        """Rename a group without changing its members."""
        existing = _get_group_or_404(group_id)
        record = store.upsert_group(
            GroupRecord(id=group_id, name=body.name, user_ids=list(existing.user_ids))
        )
        return UserGroupView.from_record(record)

    @router.post("/admin/user-group/{group_id}/add-users")
    def add_users_to_group(
        group_id: str,
        body: AddUsersToGroupRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> UserGroupView:
        """Add users to an existing group (idempotent)."""
        _get_group_or_404(group_id)
        for user_id in body.user_ids:
            store.add_user_to_group(user_id, group_id)
        return UserGroupView.from_record(store.get_group(group_id))  # type: ignore[arg-type]

    @router.post("/admin/user-group/{group_id}/remove-users")
    def remove_users_from_group(
        group_id: str,
        body: RemoveUsersFromGroupRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> UserGroupView:
        """Remove users from a group (idempotent)."""
        _get_group_or_404(group_id)
        for user_id in body.user_ids:
            store.remove_user_from_group(user_id, group_id)
        return UserGroupView.from_record(store.get_group(group_id))  # type: ignore[arg-type]

    @router.delete(
        "/admin/user-group/{group_id}",
        status_code=204,
        response_class=Response,
        response_model=None,
    )
    def delete_user_group(
        group_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> Response:
        """Delete a group and all its memberships."""
        if not store.delete_group(group_id):
            raise HTTPException(
                status_code=404, detail=f"Group {group_id!r} not found."
            )
        logger.info("Deleted group %s.", group_id)
        return Response(status_code=204)

    @router.get("/user-groups")
    def list_my_groups(
        user: AuthenticatedUser = Depends(_require_user),
    ) -> list[UserGroupView]:
        """Return groups the authenticated user belongs to."""
        return [
            UserGroupView.from_record(g) for g in store.list_groups_for_user(user.id)
        ]

    return router


__all__ = ["create_user_group_router"]
