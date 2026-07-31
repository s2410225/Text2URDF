"""Assemble URDF robot descriptions from a pre-designed module catalog.

The system is split into layers with a strict boundary (see the design spec):

* ``catalog``   -- module definitions loaded from ``modules/*/module.yaml``
* ``ir``        -- the ``AssemblySpec`` intermediate representation
* ``validation``-- deterministic, LLM-free checks returning ``ValidationError`` lists
* ``builder``   -- deterministic expansion of an ``AssemblySpec`` into URDF XML

No layer other than ``catalog.loader`` performs file I/O, and no layer in this
package performs network access.
"""

__version__ = "0.1.0"
