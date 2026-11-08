# DEPRECATED: this was an early draft of the Hindsight memory integration.
# It is unused (not imported anywhere) and contained a real bug (a typo'd
# variable in store_conversation that raised NameError if ever called),
# plus a module-level client created at import time without the
# event-loop-safety handling that src/memory/hindsight_memory.py has.
#
# The live integration is src/memory/hindsight_memory.py
# (recall_memory / store_memory / initialize_customer), which is what
# src/agent/nodes.py actually imports and uses.
#
# This file is kept only so project history isn't silently deleted;
# it can be safely removed from the repo.
