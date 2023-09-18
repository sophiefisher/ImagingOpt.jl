function G(ugrid, fftPSFs, weight, lbfreq, ubfreq, plan)
    #ytemp = real.(convolve(ugrid, fftPSFs, plan))
    #multiply by quadrature weight
    (real.(convolve(ugrid, fftPSFs, plan)) .* weight .* (ubfreq .- lbfreq) )[:]
end

function Gtranspose(ugrid, fftPSFs, weight, lbfreq, ubfreq, plan)
    #ytemp = real.(convolveT(ugrid, fftPSFs, plan))
    #multiply ytemp by quadrature weight
    (real.(convolveT(ugrid, fftPSFs, plan)) .* weight .* (ubfreq .- lbfreq) )[:]
end



#=
struct Gop <: LinearMap{Float64}
    fftPSFs::Matrix{ComplexF64}
    objL::Int
    imgL::Int
    nF::Int
    iF::Int
    lbfreq::Float64
    ubfreq::Float64
    plan::FFTW.cFFTWPlan
end   

Base.size(G::Gop) = (G.imgL^2, G.objL^2) 
GopTranspose = LinearMaps.TransposeMap{<:Any, <:Gop} # TODO: make constant
    
function Base.:(*)(G::Gop, uflat::Vector{Float64})
    #u = reshape(uflat, (G.objL, G.objL))
    #to_y(obj_plane, kernel) = real.(convolve(obj_plane, kernel, G.plan))
    #ytemp = to_y(u, G.fftPSFs)
    
    u = reshape(uflat, (G.objL, G.objL))
    ytemp = real.(convolve(u, G.fftPSFs, G.plan))
    
    #multiply by quadrature weight
    quadrature = ChainRulesCore.ignore_derivatives( ()-> ClenshawCurtisQuadrature(G.nF) )
    y = ytemp .* quadrature.weights[G.iF] .* (G.ubfreq - G.lbfreq) 

    y[:]
end
=#