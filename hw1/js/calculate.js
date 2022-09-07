function mult4( a, b ){
    var rtn = Array( 4 );
    for ( let i = 0; i < 4; i++ ) {
        rtn[i] = a[i] * b[i];
    }
    return rtn;
}