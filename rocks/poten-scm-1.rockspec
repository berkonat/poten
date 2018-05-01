package = "poten"
version = "scm-1"

source = {
   url = "git://github.com/berkonat/poten.git",
}

description = {
   summary = "Interatomic Potential package for Torch7",
   detailed = [[
   ]],
   homepage = "https://github.com/berkonat/poten",
   license = "GPL-3.0"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
