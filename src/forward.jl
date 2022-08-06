struct Gop <: LinearMap{Float64}
    fftPSFs::AbstractArray # todo: fix
    objL::Int
    imgL::Int
    nF::Int
    iF::Int
    lbfreq::Float64
    ubfreq::Float64
    padded::AbstractArray
end   

# TODO: as usual, type better
function Gop(fftPSFs, objL, imgL, nF, iF, lbfreq, ubfreq)
    psfL = objL + imgL
    padded = Array{ComplexF64}(undef, psfL, psfL, nF)
    Gop(fftPSFs, objL, imgL, nF, iF, lbfreq, ubfreq, padded) 
end

Base.size(G::Gop) = (G.imgL^2, G.objL^2) 
GopTranspose = LinearMaps.TransposeMap{<:Any, <:Gop} # TODO: make constant
    
function Base.:(*)(G::Gop, uflat::AbstractVector)
    u = reshape(uflat, (G.objL, G.objL))
    to_y(obj_plane, kernel) = real.(convolve(obj_plane, kernel))
    y = to_y(u, G.fftPSFs)
    
    #multiply by quadrature weight
    quadrature = ClenshawCurtisQuadrature(G.nF)
    y = y .* quadrature.weights[G.iF] .* (G.ubfreq - G.lbfreq) 

    y[:]
end
