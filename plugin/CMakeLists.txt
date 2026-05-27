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
    SegmentMap.cpp
    SelectionApply.cpp
    SnapshotFingerprint.cpp
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

install(TARGETS kritasmartselect
    LIBRARY DESTINATION pykrita/smart_select/lib
)

install(FILES smartselect.json
    DESTINATION ${KRITA_PLUGIN_INSTALL_DIR}
)

install(FILES SmartSelectTool.action
    DESTINATION ${KDE_INSTALL_DATADIR}/krita/actions
)
