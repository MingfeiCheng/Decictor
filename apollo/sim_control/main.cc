#include <iostream>
#include "cyber/common/global_data.h"
#include "cyber/init.h"
#include <memory>
#include "modules/dreamview/backend/map/map_service.h"
#include "modules/sim_control/sim_control.h"
#include <unistd.h>
#include <stdlib.h>
#include "modules/dreamview/backend/common/dreamview_gflags.h"

// bazel build //modules/sim_control_standalone:main
// bazel run //modules/sim_control_standalone:main
// /apollo/bazel-bin/modules/sim_control_standalone/main
// /apollo/bazel-bin/modules/sim_control_standalone/main --flagfile=/apollo/modules/common/data/global_flagfile.txt

// nohup /apollo/bazel-bin/modules/sim_control/sim_control_main &
// pkill -f 'sim_control_main'

int float_test(const char *s) {
    char *endptr;
    errno = 0;
    strtof(s, &endptr);
    if (s == endptr) {
        return 0;
    }
    while (isspace((unsigned char) *endptr)) {
        // look past the number for junk
        endptr++;
    }
    if (*endptr) {
        return 0;
    }
    if (errno) {
        // likely under/overflow
    return 0;
    }

    return 1;
}

int main(int argc, char *argv[]) {
    apollo::cyber::Init(argv[0]);

    google::ReadFromFlagsFile("/apollo/modules/common/data/global_flagfile.txt", argv[0], true);
    // google::ParseCommandLineFlags(&argc, &argv, true);

    std::unique_ptr<apollo::dreamview::MapService> map_service_;
    std::unique_ptr<apollo::dreamview::SimControl> sim_control_;

    map_service_.reset(new apollo::dreamview::MapService());
    sim_control_.reset(new apollo::dreamview::SimControl(map_service_.get()));

//    if (argc == 3 && float_test(argv[1]) && float_test(argv[2])) {
//        double x = atof(argv[1]);
//        double y = atof(argv[2]);
//        sim_control_->Start(x, y);
//    } else {
//        sim_control_->Start();
//    }

    sim_control_->Start();
    
    apollo::cyber::WaitForShutdown();
    return 0;
}