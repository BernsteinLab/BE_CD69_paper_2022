version 1.0

workflow RNA_processing_PE {

  input {

    Array[File] fastqs_r1
    Array[File] fastqs_r2

    Float fastp_memory = 16
    Int fastp_disk_space = 100
    Int fastp_num_threads = 8
    Int fastp_num_preempt = 5

    Float star_memory = 64
    Int star_disk_space = 250
    Int star_num_threads = 16
    Int star_num_preempt = 5
    File star_index # .tar.gz star_index. hg38/gencode v38 GTF

    Float picard_memory = 64
    Int picard_disk_space = 250
    Int picard_num_threads = 16
    Int picard_num_preempt = 5
    File picard_genome_reference # .tar.gz of genome fasta, dict, index
    File picard_genotyping_map # hg38 crosscheckfingerprints map

    Float rnaseq_qc2_memory = 16
    Int rnaseq_qc2_disk_space = 200
    Int rnaseq_qc2_num_threads = 16
    Int rnaseq_qc2_num_preempt = 5
    File rnaseq_qc2_gtf

  }

  scatter (i in range(length(fastqs_r1))) {
    call fastp_PE { input: fastq_r1 = fastqs_r1[i], fastq_r2 = fastqs_r2[i],
      disk_space = fastp_disk_space, memory = fastp_memory,
      num_threads = fastp_num_threads, num_preempt = fastp_num_preempt
    }

    call STAR_genome_align_PE { input: trimmed_fastq_r1 = fastp_PE.trimmed_fastq_r1,
      trimmed_fastq_r2 = fastp_PE.trimmed_fastq_r2, index = star_index, memory = star_memory,
      disk_space = star_disk_space, num_threads = star_num_threads, num_preempt = star_num_preempt
    }

    call picard_md_crosscheck { input :
      star_bam = STAR_genome_align_PE.genome_bam, memory = picard_memory,
      disk_space = picard_disk_space, num_threads = picard_num_threads,
      num_preempt = picard_num_preempt, genome_reference = picard_genome_reference,
      genotyping_map = picard_genotyping_map # hg38 crosscheckfingerprints map
    }

    call rnaseq_qc2_PE { input: star_bam =  STAR_genome_align_PE.genome_bam, disk_space= rnaseq_qc2_disk_space,
      num_threads = rnaseq_qc2_num_threads, num_preempt = rnaseq_qc2_num_preempt,
      memory = rnaseq_qc2_memory, input_gtf = rnaseq_qc2_gtf
    }
  }


  output {
    Array[File] RNA_fastp_reports = fastp_PE.json_report_PE
    Array[File] RNA_STAR_genome_bams = STAR_genome_align_PE.genome_bam
    Array[File] RNA_STAR_logs = STAR_genome_align_PE.log_file
    Array[File] RNA_unique_bam = picard_md_crosscheck.md_bam
    Array[Float] RNA_duplicate_rate = picard_md_crosscheck.md_rate
    Array[File] RNA_md_metrics = picard_md_crosscheck.md_metrics
    Array[File] RNA_genotyping_VCF = picard_md_crosscheck.Crosscheck_VCF

    Array[File] RNA_bam_rnaseqc2_tsv = rnaseq_qc2_PE.rnaseq_qc2_tsv
    Array[Float] RNA_mapping_rate = rnaseq_qc2_PE.mapping_rate
    Array[Float] RNA_exonic_rate = rnaseq_qc2_PE.exonic_rate
    Array[Float] RNA_rrna_rate = rnaseq_qc2_PE.rrna_rate
    Array[Float] RNA_read_length = rnaseq_qc2_PE.read_length
    Array[Float] RNA_genes_detected = rnaseq_qc2_PE.genes_detected
    Array[Float] RNA_mapped_reads = rnaseq_qc2_PE.mapped_reads
    Array[Float] RNA_bias_3_prime = rnaseq_qc2_PE.bias_3_prime
    
    Array[File] RNA_trimmed_fastq_r1 = fastp_PE.trimmed_fastq_r1
    Array[File] RNA_trimmed_fastq_r2 = fastp_PE.trimmed_fastq_r2
  }
}


task fastp_PE {
  input {
    File fastq_r1
    File fastq_r2
    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory
    String srr_id = basename(fastq_r1, "_1.fastq.gz")
  }
  command {
    set -euo pipefail
    fastp \
    -i ${fastq_r1} \
    -I ${fastq_r2} \
    -o ${srr_id}_1.trimmed.fastq.gz \
    -O ${srr_id}_2.trimmed.fastq.gz \
    --json ${srr_id}.json \
    --thread ${num_threads}
  }
  output {
    File trimmed_fastq_r1 = "${srr_id}_1.trimmed.fastq.gz"
    File trimmed_fastq_r2 = "${srr_id}_2.trimmed.fastq.gz"
    File json_report_PE ="${srr_id}.json"

  }
  runtime {
    docker: "bromberglab/fastp"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}


task STAR_genome_align_PE {
  input {
    File trimmed_fastq_r1
    File trimmed_fastq_r2
    String srr_id = basename(trimmed_fastq_r1, "_1.trimmed.fastq.gz")
    File index
    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory
    String index_base_name = basename(index, ".tar.gz")
  }
  command {
    set -euo pipefail
    tar -xzvf ${index}
    chmod 777 ${index}

    STAR --runMode alignReads \
    --runThreadN ${num_threads} \
    --genomeDir ${index_base_name} \
    --twopassMode Basic \
    --outFilterMultimapNmax 20 \
    --alignSJoverhangMin 8 \
    --alignSJDBoverhangMin 1 \
    --outFilterMismatchNmax 999 \
    --outFilterMismatchNoverLmax 0.1 \
    --alignIntronMin 20 \
    --alignIntronMax 1000000 \
    --alignMatesGapMax 1000000 \
    --outFilterType BySJout \
    --outFilterScoreMinOverLread 0.33 \
    --outFilterMatchNminOverLread 0.33 \
    --limitSjdbInsertNsj 1200000 \
    --readFilesIn ${trimmed_fastq_r1} ${trimmed_fastq_r2} \
    --readFilesCommand zcat \
    --outFileNamePrefix ${srr_id} \
    --outSAMstrandField intronMotif \
    --outFilterIntronMotifs None \
    --alignSoftClipAtReferenceEnds Yes \
    --outSAMtype BAM SortedByCoordinate \
    --outSAMunmapped Within \
    --genomeLoad NoSharedMemory \
    --chimSegmentMin 15 \
    --chimJunctionOverhangMin 15 \
    --chimOutType Junctions WithinBAM SoftClip \
    --chimMainSegmentMultNmax 1 \
    --outSAMattributes NH HI AS nM NM ch \
    --outSAMattrRGline ID:rg1 SM:sm1
  }

  output {
    File genome_bam = "${srr_id}Aligned.sortedByCoord.out.bam"
    File log_file = "${srr_id}Log.final.out"
  }

  runtime {
    docker: "quay.io/biocontainers/star:2.7.9a--h9ee0642_0"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}

task picard_md_crosscheck {
  input {
    File star_bam
    String bam_basename = basename(star_bam, "Aligned.sortedByCoord.out.bam")
    File genotyping_map
    File genome_reference
    String genome_basename = basename(genome_reference, ".tar.gz")
    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory

    String lbrace = "{"
    String rbrace = "}"
  }

  command {
    set -euo pipefail

    tar -xzvf ${genome_reference}

    java -jar /usr/picard/picard.jar MarkDuplicates \
    I=${star_bam} \
    O=${bam_basename}.md.bam \
    M=${bam_basename}.marked_dup_metrics.txt \
    ASSUME_SORT_ORDER=coordinate \
    REMOVE_DUPLICATES=TRUE

    grep -A2 'picard.sam.DuplicationMetrics' ${bam_basename}.marked_dup_metrics.txt | tail -1 | rev | awk '${lbrace} print $1${rbrace}' | rev > dup_rate.txt

    java -jar /usr/picard/picard.jar BuildBamIndex I=${bam_basename}.md.bam

    java -Xmx2500m -jar /usr/picard/picard.jar ExtractFingerprint \
    I=${bam_basename}.md.bam \
    O=${bam_basename}.vcf \
    H=${genotyping_map} \
    R=${genome_basename}/${genome_basename}.fa \
    VALIDATION_STRINGENCY=LENIENT
  }

  output {
    File md_bam = "${bam_basename}.md.bam"
    File md_metrics = "${bam_basename}.marked_dup_metrics.txt"
    Float md_rate = read_float("dup_rate.txt")
    File Crosscheck_VCF = "${bam_basename}.vcf"
  }

  runtime {
    docker: "broadinstitute/picard"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }

}

task rnaseq_qc2_PE {
  input {
    File star_bam
    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory
    String bam_basename = basename(star_bam, "Aligned.sortedByCoord.out.bam")
    File input_gtf

    String lbrace = "{"
    String rbrace = "}"
  }
  command {
    set -euox pipefail
    rnaseqc --sample=${bam_basename} --coverage --detection-threshold=5 \
    ${input_gtf} ${star_bam} rnaseqc2_output

    grep 'Mapping Rate' rnaseqc2_output/${bam_basename}.metrics.tsv | grep -v 'End' | rev | awk '${lbrace} print $1${rbrace}' | rev > rnaseqc2_output/mapping_rate.txt
    grep 'Exonic Rate' rnaseqc2_output/${bam_basename}.metrics.tsv | grep -v 'End\|Quality' | rev | awk '${lbrace} print $1${rbrace}' | rev > rnaseqc2_output/exonic_rate.txt
    grep 'rRNA Rate'  rnaseqc2_output/${bam_basename}.metrics.tsv| rev | awk '${lbrace} print $1${rbrace}' | rev > rnaseqc2_output/rrna_rate.txt
    grep 'Read Length' rnaseqc2_output/${bam_basename}.metrics.tsv | rev | awk '${lbrace} print $1${rbrace}' | rev> rnaseqc2_output/read_length.txt
    grep 'Genes Detected' rnaseqc2_output/${bam_basename}.metrics.tsv | rev | awk '${lbrace} print $1${rbrace}' | rev > rnaseqc2_output/genes_detected.txt
    grep 'Mapped Reads' rnaseqc2_output/${bam_basename}.metrics.tsv | grep -v 'End' | rev | awk '${lbrace} print $1${rbrace}' | rev> rnaseqc2_output/mapped_reads.txt
    grep "Mean 3' bias" rnaseqc2_output/${bam_basename}.metrics.tsv | rev | awk '${lbrace} print $1${rbrace}' | rev > rnaseqc2_output/bias_3_prime.txt
  }
  output {
    File rnaseq_qc2_tsv = "rnaseqc2_output/${bam_basename}.metrics.tsv"
    Float mapping_rate = read_float("rnaseqc2_output/mapping_rate.txt")
    Float exonic_rate = read_float("rnaseqc2_output/exonic_rate.txt")
    Float rrna_rate = read_float("rnaseqc2_output/rrna_rate.txt")
    Float read_length = read_float("rnaseqc2_output/read_length.txt")
    Float genes_detected = read_float("rnaseqc2_output/genes_detected.txt")
    Float mapped_reads = read_float("rnaseqc2_output/mapped_reads.txt")
    Float bias_3_prime = read_float("rnaseqc2_output/bias_3_prime.txt")
  }
  runtime {
    docker: "gcr.io/broad-cga-aarong-gtex/rnaseqc:latest"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}