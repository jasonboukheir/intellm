add_executable(test_wht test_wht.cpp)
target_link_libraries(test_wht PRIVATE turboquant)

add_executable(test_rotors test_rotors.cpp)
target_link_libraries(test_rotors PRIVATE rotorquant)

add_executable(test_quantize test_quantize.cpp)
target_link_libraries(test_quantize PRIVATE kvcache_quant)
