# Copies the controller and every non-system DLL it needs into the Webots
# controller folder. Windows loads DLLs from the exe's own folder first, which
# keeps Webots' bundled MinGW/Qt runtime from shadowing ours.
# Usage: cmake -DEXE=<exe> -DDEST=<dir> -DSEARCH_DIRS=<a;b> -DQT_PLUGINS=<dir> -P deploy_controller.cmake
file(MAKE_DIRECTORY "${DEST}")
file(COPY "${EXE}" DESTINATION "${DEST}")

set(_plugin "")
if(QT_PLUGINS AND EXISTS "${QT_PLUGINS}/platforms/qwindows.dll")
    file(COPY "${QT_PLUGINS}/platforms/qwindows.dll" DESTINATION "${DEST}/platforms")
    set(_plugin "${QT_PLUGINS}/platforms/qwindows.dll")
    file(WRITE "${DEST}/qt.conf" "[Paths]\nPlugins = .\n")
endif()

file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${EXE}"
    LIBRARIES ${_plugin}
    RESOLVED_DEPENDENCIES_VAR _deps
    UNRESOLVED_DEPENDENCIES_VAR _missing
    DIRECTORIES ${SEARCH_DIRS}
    PRE_EXCLUDE_REGEXES "^api-ms-.*" "^ext-ms-.*"
    POST_EXCLUDE_REGEXES ".*[Ss]ystem32.*" ".*[Ww]ebots.*"
)
foreach(_d ${_deps})
    file(COPY "${_d}" DESTINATION "${DEST}")
endforeach()
list(LENGTH _deps _n)
message(STATUS "[deploy] ${_n} DLLs copied to ${DEST}")
if(_missing)
    message(STATUS "[deploy] resolved at runtime by Webots/system: ${_missing}")
endif()
