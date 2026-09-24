"""FastAPI dependencies.

Routes receive the container that ``create_app`` attached to ``app.state`` -- a
single import alias, so tests can swap the whole dependency graph by building
the app differently.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.container import Container


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]
