Show how to use a CMake ToolChaib file as the CMAKE_USER_MAKE_RULES_OVERRIDE.
This is useful if you have required compiler flags you need set for
both verifying the compiler works and for making all your targets.

This is really an exercise in understanding the lack of interaction between
toolchains / cross compilation and CMAKE_USER_MAKE_RULES_OVERRIDE