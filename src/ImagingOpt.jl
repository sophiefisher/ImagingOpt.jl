module ImagingOpt

    export PhysicsParams, RecoveryParams, ImagingParams, OptimizeParams, JobParams
    export prepare_incident, prepare_n2f_kernel, prepare_objects, prepare_geoms, prepare_blackbody, permittivity, prepare_surrogate, prepare_reconstruction
    export Gop
    export geoms_to_far, far_to_PSFs, PSFs_to_G, make_images, reconstruct_object, reconstruction_objective
    export get_params, test_init, test_forwardmodel_perfreq, test_forwardmodel, test_design_broadband_lens,  design_broadband_lens_objective
    export arrarr_to_multi

    using FFTW
    using UUIDs
    using Distributed
    import LinearAlgebra
    import LinearAlgebra.mul!
    using LinearMaps
    using Zygote: gradient, @adjoint, @showgrad, @nograd
    using Zygote
    using Random: randperm, randsubseq
    using Random
    using FiniteDifferences: central_fdm
    using Printf
    using Statistics
    using Setfield
    using StructTypes
    using Glob
    using JSON3
    using Dates
    using Optimisers
    using JLD2
    using FLoops
    using NaturalSort
    using Images
    using ShiftedArrays
    using Augmentor
    using TestImages
    using MAT
    using FastChebInterp
    using QuadratureRules
    using Profile
    using StatProfilerHTML
    using ProfileVega
    using WavePropagation
    using ChainRulesCore
    using StaticArrays
    using DelimitedFiles
    using JuMP
    using NLopt
    using ThreadsX
    using ChainRules
    using ChainRules: RuleConfig, HasReverseMode, rrule_via_ad, ProjectTo, NoTangent, unthunk

    include("prepare.jl")
    include("utils.jl")
    include("forward.jl")
    include("pipeline.jl")
    include("optimize.jl")

end

