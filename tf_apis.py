import tensorflow as tf
import gp_apis

def make_hash_map(input1, input2;, device0):
    @tf.custom_gradient
    def _lambda(X1, X2;):
        return make_hash_map_real(X1, X2;, device0)
    return _lambda(input1, input2;)

def make_hash_map_real(input1, input2;, device0):
    out = gp_apis.gp_make_hash_map(input1, input2;, device0)
    def grad(dZ1, dZ2;):
        return gp_apis.gp_make_hash_map(dZ1, dZ2;, device0)
    return out, grad

def query_hash_map(input1, input2, input3;, device0):
    @tf.custom_gradient
    def _lambda(X1, X2, X3;):
        return query_hash_map_real(X1, X2, X3;, device0)
    return _lambda(input1, input2, input3;)

def query_hash_map_real(input1, input2, input3;, device0):
    out = gp_apis.gp_query_hash_map(input1, input2, input3;, device0)
    def grad(dZ1, dZ2, dZ3;):
        return gp_apis.gp_query_hash_map(dZ1, dZ2, dZ3;, device0)
    return out, grad

def apply_rulebook(input1, input2, input3, input4;, device0):
    @tf.custom_gradient
    def _lambda(X1, X2, X3, X4;):
        return apply_rulebook_real(X1, X2, X3, X4;, device0)
    return _lambda(input1, input2, input3, input4;)

def apply_rulebook_real(input1, input2, input3, input4;, device0):
    out = gp_apis.gp_apply_rulebook(input1, input2, input3, input4;, device0)
    def grad(dZ1, dZ2, dZ3, dZ4;):
        return gp_apis.gp_apply_rulebook(dZ1, dZ2, dZ3, dZ4;, device0)
    return out, grad

def apply_rulebook_back_dx(input1, input2, input3, input4;, device0):
    @tf.custom_gradient
    def _lambda(X1, X2, X3, X4;):
        return apply_rulebook_back_dx_real(X1, X2, X3, X4;, device0)
    return _lambda(input1, input2, input3, input4;)

def apply_rulebook_back_dx_real(input1, input2, input3, input4;, device0):
    out = gp_apis.gp_apply_rulebook_back_dx(input1, input2, input3, input4;, device0)
    def grad(dZ1, dZ2, dZ3, dZ4;):
        return gp_apis.gp_apply_rulebook_back_dx(dZ1, dZ2, dZ3, dZ4;, device0)
    return out, grad

def apply_rulebook_back_dw(input1, input2, input3, input4;, device0):
    @tf.custom_gradient
    def _lambda(X1, X2, X3, X4;):
        return apply_rulebook_back_dw_real(X1, X2, X3, X4;, device0)
    return _lambda(input1, input2, input3, input4;)

def apply_rulebook_back_dw_real(input1, input2, input3, input4;, device0):
    out = gp_apis.gp_apply_rulebook_back_dw(input1, input2, input3, input4;, device0)
    def grad(dZ1, dZ2, dZ3, dZ4;):
        return gp_apis.gp_apply_rulebook_back_dw(dZ1, dZ2, dZ3, dZ4;, device0)
    return out, grad

