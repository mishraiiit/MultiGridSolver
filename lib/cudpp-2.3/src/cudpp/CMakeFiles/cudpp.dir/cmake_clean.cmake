file(REMOVE_RECURSE
  "../../lib/libcudpp.pdb"
  "../../lib/libcudpp.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/cudpp.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
