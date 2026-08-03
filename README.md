# Mesh

Mesh creation using:

- Octahedron method: initialization with octahedron and add new vertices from center of faces.
- Convex hull strategy: initialization with a tetrahedron and add faces to the initial volume by buiding convex hull between new vertex and volume. Iterate still get all points belonging to the volume.
