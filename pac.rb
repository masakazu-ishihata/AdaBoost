#!/usr/bin/env ruby

################################################################################
# default
################################################################################
@file = "data.txt"
@n = 5

################################################################################
# Arguments
################################################################################
require "optparse"
OptionParser.new { |opts|
  # options
  opts.on("-h","--help","Show this message") {
    puts opts
    exit
  }
  opts.on("-n [int]","--n-fold","# folds") { |f|
    @n = f.to_i
  }
  # parse
  opts.parse!(ARGV)
}

################################################################################
# problem generator
################################################################################
class MyGenerator
  #### new ####
  def initialize(n)
    @n = n # # variables
  end
end

################################################################################
# classes
################################################################################
######## dataset ########
class MyData
  #### new ####
  def initialize(file)
    @file = file
    @data = []
    @vals = []

    # all values replaced by numbers
    open(file).read.split(/\n/).each do |line|
      ary = line.split(' ')
      datum = []
      for i in 0..ary.size-1
        @vals[i] = Hash.new(nil)         if @vals[i] == nil
        @vals[i][ary[i]] = @vals[i].size if @vals[i][ary[i]] == nil
        datum.push(@vals[i][ary[i]])
      end
      @data.push(datum)
    end
  end
  attr_reader :data

  #### # values of each attributes ####
  def property
    @vals.map{|i| i.size}
  end

  #### split the dataset into n datasets ####
  def split(n)
    size = (@data.size / n.to_f).to_i # size of a sub-dataset
    data = @data.shuffle

    sets = []
    for i in 0..n-1
      sets.push(data[i*size..(i+1)*size-1])
    end
    sets
  end

  #### show ####
  def show
    puts "file = #{@file}"

    # attributes
    puts "attr = #{@vals.size}"
    for i in 0..@vals.size-1
      puts "attr #{i} = #{@vals[i].size} values:  #{@vals[i].keys.join(", ")}"
    end

    # data
    @data[0..10].each do |datum|
      puts "#{datum}"
    end
    puts "..."
  end
end

######## all learner must have the following methods ########
class MyLearner
  def initizlize
    @c = 0
    @n = 0
    @v = []
  end

  #### reset learner: arg = # values of each attributes ####
  def reset(prop)
    @c = prop.shift # # class
    @n = prop.size  # # attributes
    @v = prop       # # values of each attribute
  end

  #### show ####
  def show
  end

  #### learn from data ####
  def learn(data)
  end

  #### predict ####
  def predict(datum)
    return rand(@c)
  end
end

######## test ########
class MyTester
  #### new ####
  def initialize(file)
    @data = MyData.new(file)
    @sets = nil
  end

  #### test learners by n-fold closs validation ####
  def test(ls, n)
    @sets = @data.split(n)           # testing datasets
    afs = Array.new(ls.size){|i| 0}  # averaged f-measures of each learner

    for i in 0..n-1
      fs = test_i(ls, i)
      printf("%3d %s\n", i+1, fs.map{|f| sprintf("%10.5e", f)}.join(" "))
      for j in 0..ls.size-1
        afs[j] += fs[j]
      end
    end
    afs.map!{|f| f /= n.to_i}
    puts "Ave #{afs.map{|af| sprintf("%10.5e", af)}.join(" ")}"
  end

  #### test the i-th fold ####
  def test_i(ls, i)
    tr = @sets[i]          # training set
    te = @data.data - tr   # test set
    fs = [] # averaged f-measures

    for i in 0..ls.size-1
      # reset
      ls[i].reset(@data.property)

      # learn
      ls[i].learn(tr)

      # test
      f = test_by(ls[i], te)  # f-measure
      fs.push(f)
    end

    fs
  end

  #### test lnr by te ####
  def test_by(lnr, te)
    # test
    pred = Hash.new(0)
    te.each do |datum|
      dc = datum[0]               # true class
      da = datum[1..datum.size-1] #
      pc = lnr.predict(da)        # predicted class
      pred[[dc, pc]] += 1
    end

    # f-measures
    @c = @data.property[0]   # # classes
    ave_f = 0                # average f-measure for all c

    for c in 0..@c-1       # positive class
      # {true, false} * {positive, negative}
      tp = 0
      tn = 0
      fp = 0
      fn = 0
      pred.keys.each do |key|
        tc = key[0] # true class
        pc = key[1] # predicted class
        n = pred[key]
        tp += n if tc == c && pc == c
        tn += n if tc != c && pc != c
        fp += n if tc != c && pc == c
        fn += n if tc == c && pc != c
      end

      # recall, presision
      rec = tp / (tp + fn).to_f
      pre = tp / (tp + fp).to_f

      # f-measure
      f = 2 * rec * pre / (rec + pre)
      ave_f += f
    end
    ave_f /= @c.to_f
  end
end

#### disjunctive expression learner ####
class MyDisjLearner #< MyLearner
  def initilize
    @c = 0
    @n = 0
    @v = []
    @disj = []
  end

  #### reset ####
  def reset(prop)
    @c = prop[0]      # # class
    @n = prop.size-1  # # attributes
    @v = prop[1..@n]  # # values of each attribute
    @disj = Array.new(@c-1){ |c|
      Array.new(@n){|i| Array.new(@v[i]){|v| v}}  # disj for class c
    }
  end

  #### show ####
  def show
    puts "# classes    = #{@c}"
    puts "# attributes # #{@n}"
    for c in 0..@c-2
      puts "disj for class #{c}"
      for i in 0..@n-1
        puts "x_#{i} = #{@disj[c][i].join(" v ")}" if @disj[c][i].size > 0
      end
    end
  end

  #### learn ####
  def learn(data)
    data.each do |datum|
      dc = datum[0]               # true class
      da = datum[1..datum.size-1] # attributes

      # remove da from disj[c != dc]
      for c in 0..@c-2
        next if c == dc
        for i in 0..@n-1
          @disj[c][i] -= [da[i]]
        end
      end
    end
  end

  #### predict ####
  def predict(da)
    for c in 0..@c-2
      for i in 0..@n-1
        return c if @disj[c][i].index(da[i]) != nil
      end
    end
    return @c-1
  end
end

################################################################################
# main
################################################################################
t = MyTester.new(@file)

l0 = MyLearner.new         # random
l1 = MyDisjLearner.new     # disj learner

# 10-fold closs validation
t.test([l0, l1], @n)
