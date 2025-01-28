load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//tools/platform:build_defs.bzl", "copts_if_teleop")
load("//tools/install:install.bzl", "install", "install_files")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sim_control",
    srcs = ["sim_control.cc"],
    hdrs = [
        "sim_control.h",
        "sim_control_interface.h",
    ],
    deps = [
        "//cyber",
        "//modules/common/adapters:adapter_gflags",
        "//modules/common/util:message_util",
        "//modules/dreamview/backend/common:dreamview_gflags",
        "//modules/dreamview/backend/map:map_service",
        "//modules/map/relative_map/proto:navigation_cc_proto",
        "//modules/planning/proto:planning_cc_proto",
        "//modules/prediction/proto:prediction_obstacle_cc_proto",
        "//modules/prediction/proto:scenario_cc_proto",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "sim_control_main",
    srcs = ["main.cc"],
    deps = [
        "//modules/dreamview/backend/map:map_service",
        "//modules/sim_control:sim_control"
    ],
)

py_binary(
    name = "sim_control_test",
    srcs = ["sim_control_test.py"],
    deps = [
        "//cyber/python/cyber_py3:cyber",
        "//cyber/python/cyber_py3:cyber_time",
        "//modules/common/proto:geometry_py_pb2",
        "//modules/common/proto:header_py_pb2",
        "//modules/routing/proto:routing_py_pb2",
        "//modules/localization/proto:localization_py_pb2",
        "//modules/perception/proto:perception_obstacle_py_pb2",
    ]
)

cpplint()
