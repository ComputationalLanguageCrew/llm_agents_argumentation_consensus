""" This module provides the Argument class, which is used to represent
 an argument in a debate or discussion setting."""

from typing import Dict, List, Optional


class Argument:
     """ Represents an individual argument in a debate or discussion setting. """
    def __init__(
        self,
        id: str,
        text: str,
        creator: Optional[str] = None,
        is_target: bool = False,
    ):
        self.id = id
        self.text = text
        self.creator = creator
        self.is_target = is_target
        self.supported_by: List[str] = []
        self.opposed_by: List[str] = []
        self.labelling: Dict[str, int] = {}

    def __str__(self):
        return f"{self.id} (creator: {self.creator})"
