cmake_minimum_required(VERSION 4.4)
project(amdfq C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(amdfq SHARED
        amdfq.c
        amdfq_live.c
        amdfq_log.c
        amdfq_tail.c
        amdfq_hooks_hip.c)

target_compile_options(amdfq PRIVATE -Wall -Wextra -Wformat=2)
target_link_libraries(amdfq PRIVATE ${CMAKE_DL_LIBS})
