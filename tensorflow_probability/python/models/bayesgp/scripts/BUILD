

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)


py_library(
    name = "scripts",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":bayesiangp",
        ":bgpmodel",
        ":multibayesiangp",
        ":multibgpmodel"
        ":calibrationmodel",
        ":calibration",
        ":bgpkernels",
        ":multigpkernels",
        ":bgputils",
        ":sensitivity",
    ],
)


py_library(
    name = "bgpmodel",
    srcs = ["bgpmodel.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":bayesiangp",
        ":sensitivity",
        # matplotlib,
    ],
)


py_library(
    name = "bayesiangp",
    srcs = ["bayesiangp.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        ":bgpkernels",
        ":bgputils",
    ],
)

py_library(
    name = "multibgpmodel",
    srcs = ["multibgpmodel.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":multibayesiangp",
        ":sensitivity",
        # matplotlib,
    ],
)


py_library(
    name = "multibayesiangp",
    srcs = ["multibayesiangp.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        ":multigpkernels",
        ":bgputils",
    ],
)

py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    srcs_version = "PY2AND3",
    deps = [
      "//tensorflow_probability/python/bijectors",
      "//tensorflow_probability/python/distributions",
      ":bgpkernels",
      ":bgputils"

    ],
)

py_library(
    name = "calibrationmodel",
    srcs = ["calibrationmodel.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibration",
        ":sensitivity",
        # matplotlib
        # pandas
    ],
)

py_library(
    name = "bgpkernels",
    srcs = ["bgpkernels.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "multigpkernels",
    srcs = ["multigpkernels.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "bgputils",
    srcs = ["bgputils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_probability/python/mcmc",
    ],
)

py_library(
    name = "sensitivity",
    srcs = ["sensitivity.py"],
    srcs_version = "PY2AND3",
    deps = [
        # pyDOE,
        # matplotlib,
    ],
)
