version 1.0

workflow deeptools_bam_to_bigwig {
  input {
    Array[File] bam_files
    String sample_name
    Int binsize
    File blacklist
    String filetype = "bigwig"

    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory

  }

  call merge_and_convert {input:
    bam_files=bam_files,
    filetype=filetype,
    sample_name=sample_name,
    binsize=binsize,
    blacklist=blacklist,
    filetype=filetype,
    num_threads=num_threads,
    disk_space=disk_space,
    num_preempt=num_preempt,
    memory=memory
  }

  output {
    File merged_bam=merge_and_convert.merged_bam
    File merged_bw=merge_and_convert.merged_bw
    File merged_bam_index=merge_and_convert.merged_bam_index

  }

}


task merge_and_convert {
  input {
    Array[File] bam_files
    String sample_name
    Int binsize
    File blacklist

    String filetype


    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
  }
  command {
    samtools merge --threads ${num_threads} ${sample_name}.merged.bam ${sep=' ' bam_files}
    samtools index ${sample_name}.merged.bam
    bamCoverage -b ${sample_name}.merged.bam -o ${sample_name}.merged.bedgraph -of bedgraph -bs ${binsize} -bl ${blacklist} -p ${num_threads}

  }

  output {
    File merged_bam = "${sample_name}.merged.bam"
    File merged_bam_index = "${sample_name}.merged.bam.bai"
    File merged_bw = "${sample_name}.merged.bedgraph"
  }

  runtime {
    docker : "njaved/deeptools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}