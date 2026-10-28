//! Compile the subgroup GLSL compute shaders to SPIR-V for wgpu-py.

use shaderc::{
    CompileOptions, Compiler, EnvVersion, OptimizationLevel, ShaderKind, SpirvVersion, TargetEnv,
};
use std::fs;
use std::path::Path;

fn compile(source: &Path, destination: &Path) {
    let source_text = fs::read_to_string(source).expect("read subgroup GLSL");
    let compiler = Compiler::new().expect("initialize shaderc");
    let mut options = CompileOptions::new().expect("initialize shaderc options");
    options.set_optimization_level(OptimizationLevel::Performance);
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_1 as u32);
    options.set_target_spirv(SpirvVersion::V1_3);
    let artifact = compiler
        .compile_into_spirv(
            &source_text,
            ShaderKind::Compute,
            source.to_str().expect("UTF-8 shader path"),
            "main",
            Some(&options),
        )
        .expect("compile subgroup GLSL");
    // wgpu-native's SPIR-V frontend logs RelaxedPrecision decorations even
    // though they are only optional optimization hints. Strip those decorations
    // to keep runtime initialization silent and deterministic.
    let words = artifact.as_binary();
    let mut filtered = words[..5].to_vec();
    let mut offset = 5;
    while offset < words.len() {
        let instruction = words[offset];
        let word_count = (instruction >> 16) as usize;
        assert!(word_count > 0, "invalid zero-length SPIR-V instruction");
        assert!(
            offset + word_count <= words.len(),
            "SPIR-V instruction extends beyond the module"
        );
        let opcode = instruction & 0xffff;
        let relaxed_precision = opcode == 71 && word_count >= 3 && words[offset + 2] == 0;
        if !relaxed_precision {
            filtered.extend_from_slice(&words[offset..offset + word_count]);
        }
        offset += word_count;
    }
    let bytes: Vec<u8> = filtered
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    fs::write(destination, bytes).expect("write generated SPIR-V");
}

fn main() {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("LucasKanadeICGN project root");
    let shader_directory = project.join("python/src/lucas_kanade_icgn/shaders");
    compile(
        &shader_directory.join("icgn_subgroup.comp"),
        &shader_directory.join("icgn_subgroup.spv"),
    );
    compile(
        &shader_directory.join("subgroup_probe.comp"),
        &shader_directory.join("subgroup_probe.spv"),
    );
}
