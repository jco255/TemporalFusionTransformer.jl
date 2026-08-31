#
# Correctness Tests
#

# fatalerrors = length(ARGS) > 0 && ARGS[1] == "-f"
# quiet = length(ARGS) > 0 && ARGS[1] == "-q"

using Test

anyerrors = false

my_tests = ["nn_functional.jl", "add_norm.jl", "resample_norm.jl", "glu.jl", "grn.jl", "misc.jl"]
println("Running tests:")

for my_test in my_tests
    try
        include(my_test)
        println("\t\033[1m\033[32mPASSED\033[0m: $(my_test)")
    catch e
        global anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(my_test)")
        showerror(stdout, e, backtrace())
        println()
    end
end

if anyerrors
    throw("Tests failed")
end
