# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3/Bootstrap.cmk/cmake

# The command to remove a file.
RM = /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3/Bootstrap.cmk/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3

# Utility rule file for ContinuousMemCheck.

# Include the progress variables for this target.
include CMakeFiles/ContinuousMemCheck.dir/progress.make

CMakeFiles/ContinuousMemCheck:
	CMAKE_CTEST_COMMAND-NOTFOUND -D ContinuousMemCheck

ContinuousMemCheck: CMakeFiles/ContinuousMemCheck
ContinuousMemCheck: CMakeFiles/ContinuousMemCheck.dir/build.make

.PHONY : ContinuousMemCheck

# Rule to build all files generated by this target.
CMakeFiles/ContinuousMemCheck.dir/build: ContinuousMemCheck

.PHONY : CMakeFiles/ContinuousMemCheck.dir/build

CMakeFiles/ContinuousMemCheck.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ContinuousMemCheck.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ContinuousMemCheck.dir/clean

CMakeFiles/ContinuousMemCheck.dir/depend:
	cd /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3 /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3 /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3 /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3 /Users/Chester/Cal/2/clipper/deps/cmake-3.6.3/CMakeFiles/ContinuousMemCheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ContinuousMemCheck.dir/depend

