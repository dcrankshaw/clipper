// automatically generated by rust-bindgen, with some modifications
#![allow(dead_code)]
extern crate libc;
pub mod common;
use libc::{c_int, c_double, c_char, c_uint};
use common::Struct_feature_node;
use std::mem;


#[repr(C)]
#[derive(Copy)]
pub struct Struct_problem {
    pub l: c_int, // number of examples
    pub n: c_int, // max index
    pub y: *const c_double,
    pub x: *const *const Struct_feature_node, // array of pointers
    pub bias: c_double,
}
impl Clone for Struct_problem {
    fn clone(&self) -> Self {
        *self
    }
}
impl Default for Struct_problem {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

// pub type Enum_Unnamed1 = c_uint;
pub const L2R_LR: c_uint = 0;
pub const L2R_L2LOSS_SVC_DUAL: c_uint = 1;
pub const L2R_L2LOSS_SVC: c_uint = 2;
pub const L2R_L1LOSS_SVC_DUAL: c_uint = 3;
pub const MCSVM_CS: c_uint = 4;
pub const L1R_L2LOSS_SVC: c_uint = 5;
pub const L1R_LR: c_uint = 6;
pub const L2R_LR_DUAL: c_uint = 7;
pub const L2R_L2LOSS_SVR: c_uint = 11;
pub const L2R_L2LOSS_SVR_DUAL: c_uint = 12;
pub const L2R_L1LOSS_SVR_DUAL: c_uint = 13;

#[repr(C)]
#[derive(Copy)]
#[allow(non_snake_case)]
pub struct Struct_parameter {
    pub solver_type: c_uint,
    pub eps: c_double,
    pub C: c_double,
    pub nr_weight: c_int,
    pub weight_label: *const c_int,
    pub weight: *const c_double,
    pub p: c_double,
    pub init_sol: *const c_double,
}
impl Clone for Struct_parameter {
    fn clone(&self) -> Self {
        *self
    }
}
impl Default for Struct_parameter {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_model {
    pub param: Struct_parameter,
    pub nr_class: c_int,
    pub nr_feature: c_int,
    pub w: *const c_double,
    pub label: *const c_int,
    pub bias: c_double,
}
impl Clone for Struct_model {
    fn clone(&self) -> Self {
        *self
    }
}
impl Default for Struct_model {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

// #[link(name = "linear")]
extern "C" {
    pub fn train(prob: *const Struct_problem,
                 param: *const Struct_parameter)
                 -> *const Struct_model;
    pub fn cross_validation(prob: *const Struct_problem,
                            param: *const Struct_parameter,
                            nr_fold: c_int,
                            target: *const c_double)
                            -> ();
    pub fn find_parameter_C(prob: *const Struct_problem,
                            param: *const Struct_parameter,
                            nr_fold: c_int,
                            start_C: c_double,
                            max_C: c_double,
                            best_C: *const c_double,
                            best_rate: *const c_double)
                            -> ();
    pub fn predict_values(model_: *const Struct_model,
                          x: *const Struct_feature_node,
                          dec_values: *const c_double)
                          -> c_double;
    pub fn predict(model_: *const Struct_model, x: *const Struct_feature_node) -> c_double;
    pub fn predict_probability(model_: *const Struct_model,
                               x: *const Struct_feature_node,
                               prob_estimates: *const c_double)
                               -> c_double;
    pub fn save_model(model_file_name: *const c_char, model_: *const Struct_model) -> c_int;
    pub fn load_model(model_file_name: *const c_char) -> *const Struct_model;
    pub fn get_nr_feature(model_: *const Struct_model) -> c_int;
    pub fn get_nr_class(model_: *const Struct_model) -> c_int;
    pub fn get_labels(model_: *const Struct_model, label: *const c_int) -> ();
    pub fn get_decfun_coef(model_: *const Struct_model,
                           feat_idx: c_int,
                           label_idx: c_int)
                           -> c_double;
    pub fn get_decfun_bias(model_: *const Struct_model, label_idx: c_int) -> c_double;
    pub fn free_model_content(model_ptr: *const Struct_model) -> ();
    pub fn free_and_destroy_model(model_ptr_ptr: *const *const Struct_model) -> ();
    pub fn destroy_param(param: *mut Struct_parameter) -> ();
    pub fn check_parameter(prob: *const Struct_problem,
                           param: *const Struct_parameter)
                           -> *const c_char;
    pub fn check_probability_model(model: *const Struct_model) -> c_int;
    pub fn check_regression_model(model: *const Struct_model) -> c_int;
}
