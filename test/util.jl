
function transpose3(x)
    @assert ndims(x)==3
    l,m,n = size(x)
    [x[i,j,k] for k in 1:n, j in 1:m, i in 1:l]
end