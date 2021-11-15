SET(TARGET "benchmark_03")

SET(TARGET_SRC ${TARGET}.cu )

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

FIND_PACKAGE(deal.II 10.0.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_P4EST OR NOT DEAL_II_WITH_CUDA) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_CUDA = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
    DEAL_II_WITH_MPI = ${DEAL_II_WITH_MPI}
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_CUDA = ${DEAL_II_WITH_CUDA}
This conflicts with the requirements."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

