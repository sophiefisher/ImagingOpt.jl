struct Gop <: LinearMap{Float64}
    fftPSFs::Matrix{ComplexF64}
    objL::Int
    imgL::Int
    nF::Int
    iF::Int
    lbfreq::Float64
    ubfreq::Float64
    plan::FFTW.cFFTWPlan
    padded::Array{ComplexF64, 3}
end   

# TODO: as usual, type better
function Gop(fftPSFs::Matrix{ComplexF64}, objL, imgL, nF, iF, lbfreq, ubfreq, plan)
    psfL = objL + imgL
    padded = Array{ComplexF64}(undef, psfL, psfL, nF) #if the dimension of this changes, change type of padded in Gop structure
    Gop(fftPSFs, objL, imgL, nF, iF, lbfreq, ubfreq, plan, padded) 
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
