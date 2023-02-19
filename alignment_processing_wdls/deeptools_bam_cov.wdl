version 1.0

workflow deeptools_bam_to_bigwig {
  input {
    Array[File] bam_files
    String sample_name
    Float scalefactor
    Int binsize
    File blacklist
    String filetype = "bigwig"
    String genomesize = "2747877777"

    String normtype = "RPGC"

    Int smoothlength=100

    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory

  }

  call merge_and_convert {input:
    bam_files=bam_files,
    filetype=filetype,
    sample_name=sample_name,
    scalefactor=scalefactor,
    binsize=binsize,
    blacklist=blacklist,
    filetype=filetype,
    genomesize=genomesize,
    normtype=normtype,
    smoothlength=smoothlength,
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
    Float scalefactor
    Int binsize
    File blacklist

    String filetype
    
    String genomesize

    String normtype

    Int smoothlength

    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
  }
  command {
    samtools merge --threads ${num_threads} ${sample_name}.merged.bam ${sep=' ' bam_files}
    samtools index ${sample_name}.merged.bam
    bamCoverage -b ${sample_name}.merged.bam -o ${sample_name}.merged.bw -of ${filetype} --scaleFactor ${scalefactor} -bs ${binsize} -bl ${blacklist} -p ${num_threads} --effectiveGenomeSize ${genomesize} --normalizeUsing ${normtype} --smoothLength ${smoothlength}

  }

  output {
    File merged_bam = "${sample_name}.merged.bam"
    File merged_bam_index = "${sample_name}.merged.bam.bai"
    File merged_bw = "${sample_name}.merged.bw"
  }

  runtime {
    docker : "njaved/deeptools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}