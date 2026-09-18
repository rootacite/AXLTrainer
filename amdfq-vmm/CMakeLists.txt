cmake_minimum_required(VERSION 4.4)
project(amdfq-vmm C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(amdfq-vmm SHARED
        amdfq_vmm.c
        amdfq_hooks_hip.c)

target_compile_options(amdfq-vmm PRIVATE -Wall -Wextra -Wformat=2)
target_link_libraries(amdfq-vmm PRIVATE ${CMAKE_DL_LIBS})
