if(NOT COMMAND kis_add_library)
    message(FATAL_ERROR
        "krita-smart-select must be built from a Krita 5.3.1 source/build tree "
        "where Krita's CMake plugin helpers are available. Set KRITA_SRC/KRITA_BUILD "
        "or add this directory to a Krita 5.3.1 build.")
endif()

set(kritasmartselect_SOURCES
    SmartSelectPlugin.cpp
    SmartSelectRuntimeConfig.cpp
    SmartSelectTool.cpp
    LassoRegionMask.cpp
    NpyWriter.cpp
    SelectionApply.cpp
    SelectionMask.cpp
    SmartSelectLog.cpp
    SnapshotCapture.cpp
    SnapshotFingerprint.cpp
    SnapshotImage.cpp
    WorkerClient.cpp
    ${CMAKE_SOURCE_DIR}/plugins/tools/selectiontools/kis_selection_modifier_mapper.cc
)

qt_add_resources(kritasmartselect_SOURCES icons/smartselect.qrc)

kis_add_library(kritasmartselect MODULE ${kritasmartselect_SOURCES})
generate_export_header(kritasmartselect BASE_NAME kritasmartselect)

target_compile_features(kritasmartselect PRIVATE cxx_std_20)
target_include_directories(kritasmartselect PRIVATE
    .
    ${CMAKE_SOURCE_DIR}/plugins/tools/selectiontools
)

target_link_libraries(kritasmartselect PRIVATE
    kritaui
    kritabasicflakes
    kritaimage
)

option(SMART_SELECT_BUILD_TESTS "Build Krita Smart Select code tests" OFF)
if(SMART_SELECT_BUILD_TESTS)
    enable_testing()
    find_package(Qt5 REQUIRED COMPONENTS Core Gui Test)

    add_executable(smartselect_core_tests
        tests/SmartSelectCoreTest.cpp
        LassoRegionMask.cpp
        NpyWriter.cpp
        SelectionMask.cpp
        SmartSelectLog.cpp
        SnapshotFingerprint.cpp
        SnapshotImage.cpp
    )
    target_include_directories(smartselect_core_tests PRIVATE .)
    target_link_libraries(smartselect_core_tests PRIVATE
        Qt5::Core
        Qt5::Gui
        Qt5::Test
    )
    add_test(NAME smartselect_core_tests COMMAND smartselect_core_tests)
endif()

install(TARGETS kritasmartselect
    LIBRARY DESTINATION pykrita/smart_select/lib
)

install(FILES smartselect.json
    DESTINATION ${KRITA_PLUGIN_INSTALL_DIR}
)

install(FILES SmartSelectTool.action
    DESTINATION ${KDE_INSTALL_DATADIR}/krita/actions
)
