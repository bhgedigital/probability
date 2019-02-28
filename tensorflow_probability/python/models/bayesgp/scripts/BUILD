

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
        ":calibrationmodel",
        ":calibration",
        ":bgpkernels",
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
    srcs = ["calibration_model.py"],
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
    name = "bgputils",
    srcs = ["bgp_utils.py"],
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
