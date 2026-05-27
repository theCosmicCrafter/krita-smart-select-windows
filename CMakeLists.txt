cmake_minimum_required(VERSION 3.16)

project(krita-smart-select LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(plugin)

install(FILES python/smart_select.desktop
    DESTINATION pykrita
)

install(DIRECTORY python/smart_select/
    DESTINATION pykrita/smart_select
    FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY worker/
    DESTINATION pykrita/smart_select/worker
    FILES_MATCHING PATTERN "*.py"
)
