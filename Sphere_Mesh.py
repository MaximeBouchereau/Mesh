import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

# INITIALIZATION
VERTC = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]])

EDGES = [(VERTC[0, :], VERTC[1, :]), (VERTC[0, :], VERTC[2, :]), (VERTC[0, :], VERTC[3, :]), (VERTC[0, :], VERTC[4, :]),
         (VERTC[1, :], VERTC[2, :]), (VERTC[2, :], VERTC[3, :]), (VERTC[3, :], VERTC[4, :]), (VERTC[4, :], VERTC[1, :]),
         (VERTC[5, :], VERTC[1, :]), (VERTC[5, :], VERTC[2, :]), (VERTC[5, :], VERTC[3, :]), (VERTC[5, :], VERTC[4, :])]

FACES = [(VERTC[0, :], VERTC[1, :], VERTC[2, :]), (VERTC[0, :], VERTC[2, :], VERTC[3, :]), (VERTC[0, :], VERTC[3, :], VERTC[4, :]), (VERTC[0, :], VERTC[4, :], VERTC[1, :]),
         (VERTC[5, :], VERTC[1, :], VERTC[2, :]), (VERTC[5, :], VERTC[2, :], VERTC[3, :]), (VERTC[5, :], VERTC[3, :], VERTC[4, :]), (VERTC[5, :], VERTC[4, :], VERTC[1, :])]

# VERTC = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
#
# EDGES = [(VERTC[0, :], VERTC[1, :]), (VERTC[1, :], VERTC[2, :]), (VERTC[2, :], VERTC[0, :])]
#
# FACES = [(VERTC[0, :], VERTC[1, :], VERTC[2, :])]

class Tools:
    """Class for tools"""

    @staticmethod
    def Cart_Eq_Plan(A,B,C):
        """Gives cartesian equation of plan with points A, B and C, under form ax+by+cz+1=0.

        Inputs:
        - A, B, C: Arrays of shape (3,) - Points defining plan."""

        XX = np.concatenate((A.reshape(1,3),B.reshape(1,3),C.reshape(1,3)), axis=0)

        a,b,c = -np.linalg.solve(XX, np.ones(3,))

        return a,b,c

    @staticmethod
    def Test_Convex(x0, x1, X):
        """Tests every vertex among list of vertices X in order to get a correct mesh w.r.t. vertices a and b

        Inputs:
        - x0: Array of shape (3,) - First vertex
        - x1: Array of shape (3,) - Second vertex
        - X: List of arrays of shape (3,) - Other Mesh vertices"""

        for x in X:
            Y = [y for y in X if not np.allclose(y, x)]
            XX = np.concatenate([x.reshape(1, 3) for x in Y], axis=0)
            a, b, c = Tools.Cart_Eq_Plan(x0, x1, x)

            Plan = a * XX[:, 0] + b * XX[:, 1] + c * XX[:, 2] + np.ones_like(XX[:, 0])

            const_sign_plan = np.all(Plan > 0) or np.all(Plan < 0)

            if const_sign_plan:
                if not isinstance(x, np.ndarray) or x.shape != (3,):
                    print("PROBLEME Test_Convex renvoie :", x)
                    raise RuntimeError
                return x

    @staticmethod
    def same_edge(e1, e2):
        return (
                (np.allclose(e1[0], e2[0]) and np.allclose(e1[1], e2[1]))
                or
                (np.allclose(e1[0], e2[1]) and np.allclose(e1[1], e2[0]))
        )

    @staticmethod
    def same_face(f1, f2):
        pts1 = [f1[0], f1[1], f1[2]]
        pts2 = [f2[0], f2[1], f2[2]]

        return all(
            any(np.allclose(p1, p2) for p2 in pts2)
            for p1 in pts1
        )

    @staticmethod
    def Is_visible(f, p):
        """Checks whether a vertex is visible or not from a face.
        Inputs:
        - f: Tuple of three arrays of shape (3,) - Face
        - p: Array of shape (3,) -  Vertex
        """

        f1, f2, f3 = f
        n = np.cross(f2 - f1, f3 - f1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.quiver(0, 0, 0,  n[0], n[1], n[2], length=1, normalize=False)
        ax.quiver(0, 0, 0,  p[0], p[1], p[2], length=1, normalize=False)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        plt.show()


        return np.dot(n, p) > 0

    @staticmethod
    def Orient_face(f):
        f1, f2, f3 = f

        normal = np.cross(f2 - f1, f3 - f1)
        center = (f1 + f2 + f3) / 3

        if np.dot(normal, center) < 0:
            return (f1, f3, f2)
        else:
            return f

    @staticmethod
    def View_Face(p, f, y):

        f1, f2, f3 = f
        # a, b, c = Tools.Cart_Eq_Plan(f1, f2, f3)
        #
        # yy = y
        # Plan_y = a * yy[0] + b * yy[1] + c * yy[2] + np.ones_like(yy[0])
        # Plan_p = a * p[0] + b * p[1] + c * p[2] + np.ones_like(p[0])

        n = np.cross(f2 - f1, f3 - f1)
        d = -np.dot(n, f1)
        Plan_p = np.dot(n, p) + d
        Plan_y = np.dot(n, y) + d

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        #
        # triangle = Poly3DCollection([[f1, f2, f3]], alpha=0.5)
        # triangle.set_facecolor("cyan")
        # triangle.set_edgecolor("k")
        #
        # ax.add_collection3d(triangle)
        #
        # ax.scatter(p[0], p[1], p[2], color="red", s=50)
        # ax.scatter(y[0], y[1], y[2], color="red", s=50)
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])
        # ax.set_aspect("equal")
        # plt.show()

        return Plan_y * Plan_p < 0


def Mesh_Oct(N_iter):
    """Creates a mesh with regular octahedron as initialization
    Smaller trianges are creates from larger triangles by adding
    verticies at middle of each triangle edge.

    Inputs:
    - N_iter: Int - Number of iterations.

    Returns list of faces and VERTICES.
    """

    # INITIALIZATION

    VERTC = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]])

    FACES = [(VERTC[0, :], VERTC[1, :], VERTC[2, :]), (VERTC[0, :], VERTC[2, :], VERTC[3, :]),
             (VERTC[0, :], VERTC[3, :], VERTC[4, :]), (VERTC[0, :], VERTC[4, :], VERTC[1, :]),
             (VERTC[5, :], VERTC[1, :], VERTC[2, :]), (VERTC[5, :], VERTC[2, :], VERTC[3, :]),
             (VERTC[5, :], VERTC[3, :], VERTC[4, :]), (VERTC[5, :], VERTC[4, :], VERTC[1, :])]

    # ITERATIONS

    for n in range(N_iter):
        print("n = " + str(n + 1) + " / " + str(N_iter), end="\r")
        FACES_new, VERTC_new = [], VERTC
        for F in FACES:
            A, B, C = F
            AB = (A + B) / np.linalg.norm(A + B)
            BC = (B + C) / np.linalg.norm(B + C)
            CA = (C + A) / np.linalg.norm(C + A)
            FACES_new += [(A, AB, CA), (B, BC, AB), (C, CA, BC), (AB, BC, CA)]
            VERTC_new = np.concatenate((VERTC_new, AB.reshape(1, 3), BC.reshape(1, 3), CA.reshape(1, 3)), axis=0)
        FACES = FACES_new
        VERTC = VERTC_new

    return FACES, VERTC, N_iter

def Mesh_Poisson(K, r_min):
    """Creates a mesh on unit sphere with Poisson Disk Sampling.
    Remove points whose are too close eachother.

    Inputs:
    - K: Int - Number of points.
    - r_min: Positive float - Minimal distance between points.
    """

    x = np.random.normal(size=(K, 3))
    VERTC = x / np.linalg.norm(x, axis=1).reshape(K,1)

    ONES = np.ones((1, K))
    NORM = np.linalg.norm(VERTC, axis=1).reshape(K, 1)

    NORM = NORM @ ONES + (NORM @ ONES).T - 2 * VERTC @ VERTC.T

    IS_FAR = (NORM > r_min ** 2)
    np.fill_diagonal(IS_FAR, True)
    IS_FAR = np.prod(IS_FAR, axis=0)
    idx_far = np.where(IS_FAR == 1)[0]

    print(" > Number of points: " +  str(len(idx_far)) + " / " + str(K))

    if len(idx_far) < 3:
        raise RuntimeError("Points are to closed each other for specified distance.")

    VERTC = VERTC[idx_far, :]
    VERTC = VERTC[::-1, :]
    VERTC = np.concatenate((VERTC, np.zeros((VERTC.shape[0], 1))), axis=1)
    
    print(" > Faces construction")
    FACES = []
    EDGES = []
    EDGES_BOUND = []
    X = list(VERTC[:, :3]) # REMAINING VERTICES

    # Proximity matrix between vertices
    ONES = np.ones((1, VERTC.shape[0]))
    NORM = np.linalg.norm(VERTC[:, :3], axis=1).reshape(VERTC.shape[0], 1)
    NORM = NORM @ ONES + (NORM @ ONES).T - 2 * VERTC[:, :3] @ VERTC[:, :3].T

    np.fill_diagonal(NORM, 100)

    # plt.imshow(NORM, cmap="gray"); plt.show()

    # IS_CLOSE = (NORM < r_prox ** 2)
    # np.fill_diagonal(IS_CLOSE, False)
    # print(np.sum(IS_CLOSE[0, :]))

    # INITIALIZATION: FIND FIRST TRIANGLE TO CONVEX HULL
    # i0 = 0
    # i0 = np.where(VERTC[:, 2] == np.min(VERTC[:, 2]))[0][0]
    # VERTC[i0, 3] = 1
    # u0 = VERTC[i0, :3]
    # i1 = np.where(NORM[i0, :] == np.min(NORM[i0, :]))
    # VERTC[i1, 3] = 1
    # u1 = VERTC[i1, :3].squeeze()
    # idx_free = np.where(VERTC[:, 3] == 0)[0]
    # EDGES.append((u0, u1))
    # EDGES_BOUND.append((u0, u1))
    #
    #
    # Y = [x for x in X if not (np.allclose(x, u0) or np.allclose(x, u1))]
    # Z = [x for x in X if not (np.allclose(x, u0) or np.allclose(x, u1))]
    # u = Tools.Test_Convex(u0, u1, Y)
    # FACES.append((u0, u1, u))
    # EDGES_BOUND.append((u0, u))
    # EDGES_BOUND.append((u1, u))
    # Z = [x for x in Z if not np.allclose(x, u)]

    # while (len(X) > 1 and len(EDGES_BOUND) < 10):
    # while len(EDGES_BOUND) < 20:
    #     print(len(EDGES_BOUND))
    #     e = random.choice(EDGES_BOUND)
    #     e0, e1 = e
    #     Y = [x for x in X if not (np.allclose(x, e0) or np.allclose(x, e1))]
    #     Z = [x for x in Z if not (np.allclose(x, e0) or np.allclose(x, e1))]
    #     u = Tools.Test_Convex(e0, e1, Y)
    #
    #     Z = [x for x in Z if not (np.allclose(x, u))]
    #     if not any(np.allclose(x, u) for x in Z):
    #         u = Tools.Test_Convex(e0, e1, Y)
    #
    #
    #     EDGES_BOUND = [x for x in EDGES_BOUND if not Tools.same_edge(e,x)]
    #     # EDGES_BOUND.extend([(e0, u), (e1, u)])
    #     for new_edge in [(e0, u), (e1, u)]:
    #         if not any(Tools.same_edge(new_edge, e) for e in EDGES_BOUND):
    #             EDGES_BOUND.append(new_edge)
    #
    #     if not any(Tools.same_face((e0, e1, u), f) for f in FACES):
    #         FACES.append((e0, e1, u))
    #     # print(len(FACES))



    ### INITIALIZATION: TETHRAEDRON
    X = list(VERTC[:, :3]) # List of vertices
    x0, x1, x2, x3 = X[0:4]
    FACES = [(x0, x1, x2), (x1, x2, x3), (x0, x2, x3), (x0, x1, x3)]
    FACES = [Tools.Orient_face(f) for f in FACES]

    X = [x for x in X if not (np.allclose(x, x0) or np.allclose(x, x1) or np.allclose(x, x2) or np.allclose(x, x3))]
    Y = [x0, x1, x2, x3]

    ### BUILD OTHER FACES
    while len(X) > 0:
        x = random.choice(X)
        FACES_vis = []
        for f in FACES:
            Y_other = [y for y in Y if not (np.allclose(y, f[0]) or np.allclose(y, f[1]) or np.allclose(y, f[2]))]
            # print([Tools.View_Face(x, f, y) for y in Y_other])
            # if all(Tools.View_Face(x, f, y) for y in Y_other):
            y = random.choice(Y_other)
            if Tools.View_Face(x, f, y):
                FACES_vis.append(f)

        # print("f vis:", len(FACES_vis))
        # FIND HORIZON
        EDGES = []

        for f in FACES_vis:
            f0, f1, f2 = f
            EDGES.extend([(f0, f1), (f1, f2), (f2, f0)])

        # Keep only edges appearing once
        HORIZON = []

        for e in EDGES:
            count = 0
            for ee in EDGES:
                if Tools.same_edge(e, ee):
                    count += 1

            if count == 1:
                HORIZON.append(e)

        # REMOVE VISIBLE FACES
        FACES = [f for f in FACES if not any(Tools.same_face(f, fv) for fv in FACES_vis)]

        # CREATE NEW FACES
        for e in HORIZON:
            e0, e1 = e
            FACES.append(Tools.Orient_face((e0, e1, x)))

        X = [xx for xx in X if not np.allclose(xx, x)]
        Y.append(x)

        print(" > Remaining points:", len(X), "  ", end="\r")
        # print("f:", len(FACES))

    # for i2 in idx_free:
    #     u2 = VERTC[i2, :3]
    #     U = np.array([u0, u1, u2])
    #     A = np.linalg.solve(U, -np.ones(3))
    #     idx_free_new = idx_free[idx_free != i2]
    #     H = VERTC[idx_free_new, :3] @ A.reshape(3, 1) + np.ones_like(idx_free_new)
    #     # print(np.max(np.sign(H)) - np.min(np.sign(H)))
    #     if np.max(np.sign(H)) - np.min(np.sign(H)) < 0.5:
    #         idx_free = idx_free_new
    #         FACES.append((u0, u1, u2))
    #         EDGES += [(u1, u2), (u2, u0)]
    #         VERTC[i1, 3] = 1
    #         VERTC[i2, 3] = 1
    #         # break
    # while np.prod(VERTC[:, 3], axis=0) == 0:
    #     u0 = VERTC[i0, :3]
    #     i1 = np.where(NORM[i0, :] == np.min(NORM[i0, :]))
    #     VERTC[i1, 3] = 1
    #     u1 = VERTC[i1, :3]
    #     idx_free = np.where(VERTC[:, 3] == 0)[0]
    #     EDGES.append((u0, u1))
    #     for i2 in list(idx_free):
    #         u2 = VERTC[i2, :3]
    #         U = np.array([u0, u1[0][0], u2])
    #         A = np.linalg.solve(U, -np.ones(3))
    #         idx_free_new = idx_free[idx_free != i2]
    #         H =  VERTC[idx_free_new, :3] @ A.reshape(3,1) + np.ones_like(idx_free_new)
    #         if np.max(np.sign(H)) - np.min(np.sign(H)) < 0.5:
    #             idx_free = idx_free_new
    #             # FACES.append((v0, v1, v2))
    #             FACES.append((u0, u1, u2))
    #             break


    # FACES = [(VERTC[0, :3], VERTC[1, :3], VERTC[2, :3])]


    return FACES, VERTC[:, :3], 1

def Plot(MESH):
    """Plots a mesh on a sphere.
    Inputs:
    - MESH: Tuple of len 2 of the form (FACES, VERTC, N_iter) where:
        > FACES: List of tuples (a_k,b_k,c_k) where a_k, b_k, c_k are arrayes of shape (3,) -  List of faces.
        > VERTC: Array of shape (N,3) - Array whom lines are coordinates of verticies.
        > N_iter: Int - Number of iterations"""

    FACES, VERTC, N_iter = MESH

    L = 1

    # PLOT POINTS
    colors = np.random.randint(low=1, high=100, size=(VERTC.shape[0],))
    colors = np.linspace(start=1, stop=100, num=VERTC.shape[0])

    ax = plt.figure().add_subplot(projection='3d')
    VERTC_sort = VERTC[VERTC[:, 2].argsort()]
    xx, yy, zz = VERTC_sort[:, 0], VERTC_sort[:, 1], VERTC_sort[:, 2]
    ax.set_title("$N = $ "+str(N_iter))
    ax.scatter(xx, yy, zz, c=colors, depthshade=0, cmap="rainbow")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_aspect("equal")
    ax.grid()

    # PLOT MESH
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title("$N = $ "+str(N_iter))
    FACES_sort = sorted(FACES, key=lambda f: f[0][2])
    values = np.linspace(0, 1, len(FACES_sort))  # One value per triangle
    cmap = plt.cm.jet
    colors = cmap(values)
    poly = Poly3DCollection(FACES_sort, alpha=0.8)
    poly.set_facecolor(colors)      # Face color
    poly.set_edgecolor('black')     # Edge color
    poly.set_linewidth(1)           # Edge thickness
    ax.add_collection3d(poly)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_aspect("equal")
    ax.grid()
    plt.show()

    return None

def Plot_Sphere(p = 2, K = 1000, size = [1, 1, 1]):
    """Plots points randomly along a sphere.

    Inputs:
    - p: Float => 1 or inf - Topology, index of the norm. Default: 2 (Euclidian norm)
    - K: Int - Number of points. Default: 1000.
    - size: List of positive floats - Dilatation w.r.t. directions x, y and z. Default: [1, 1, 1].
    """

    X = np.random.uniform(low=-10, high=10, size=(3, K))
    # NORM = (X[0, :] ** 2 + X[1, :] ** 2 + X[2, :] ** 2) ** 0.5
    # NORM = np.abs(X[0, :]) + np.abs(X[1, :]) + np.abs(X[2, :])
    if p < np.infty:
        NORM = (np.abs(X[0, :] / size[0]) ** p + np.abs(X[1, :] / size[1]) ** p + np.abs(X[2, :] / size[2]) ** p) ** (1 / p)
    if p == np.infty:
        NORM = np.max(np.abs(np.concatenate((X[0, :].reshape(1, K) / size[0], X[1, :].reshape(1, K) / size[1], X[2, :].reshape(1, K) / size[2]), axis=0)), axis=0)
    VERTC = (X / NORM).T
    colors = np.linspace(start=1, stop=100, num=VERTC.shape[0])
    ax = plt.figure().add_subplot(projection='3d')
    VERTC_sort = VERTC[VERTC[:, 2].argsort()]
    xx, yy, zz = VERTC_sort[:, 0], VERTC_sort[:, 1], VERTC_sort[:, 2]
    ax.scatter(xx, yy, zz, c=colors, depthshade=0, cmap="rainbow")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_xlim(1.5 * VERTC_sort[:, 0].min(), 1.5 * VERTC_sort[:, 0].max())
    ax.set_ylim(1.5 * VERTC_sort[:, 1].min(), 1.5 * VERTC_sort[:, 1].max())
    ax.set_zlim(1.5 * VERTC_sort[:, 2].min(), 1.5 * VERTC_sort[:, 2].max())
    ax.set_aspect("equal")
    ax.grid()
    plt.show()

    return None