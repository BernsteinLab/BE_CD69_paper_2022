version 1.0

workflow run_enformer {
  input {
    String target_name
    String interval

    File fasta
    File peaks
    File script
    String indices
    Int target_index

    Float memory = 4
    Int disk_space = 30
    Int num_threads = 2
    Int num_preempt = 5
  }

  call enformer { input :
    target_name=target_name,
    script=script,
    peaks=peaks,
    interval=interval,
    fasta=fasta,
    indices=indices,
    target_index=target_index,
    disk_space=disk_space,
    num_threads=num_threads,
    num_preempt=num_preempt,
    memory=memory
  }

  output {
    File pooled=enformer.pooled
    File all=enformer.all
    File matrix=enformer.matrix
    File sequence=enformer.sequence
    File gradient=enformer.gradient
    File peaks_matrix=enformer.peaks_matrix
    File peaks_grad=enformer.peaks_grad
    File peaks_null=enformer.peaks_null
    File peaks_seq=enformer.peaks_seq
  }
}

task enformer {
  input {
    String target_name
    String interval

    File fasta
    File script
    String indices
    Int target_index
    
    File peaks

    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory
  }

  command {
    set -euo pipefail

    python3 ${script} ${fasta} ${interval} ${indices} ${target_index} ${target_name} ${peaks}

  }

  output {
    File pooled = "${target_name}.pooled.bedGraph"
    File all = "${target_name}.all.bedGraph"
    File matrix = "${target_name}.basescores.out"
    File sequence = "${target_name}.sequence"
    File gradient = "${target_name}.gradient.out"
    
    File peaks_matrix = "${target_name}.peaks.basescores.npy"
    File peaks_grad = "${target_name}.peaks.gradient.npy"
    File peaks_null = "${target_name}.peaks.null.npy"
    File peaks_seq = "${target_name}.peaks.sequence.npy"
  }

  runtime {
    docker: "njaved/enformer"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}