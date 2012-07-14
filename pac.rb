#!/usr/bin/env ruby

################################################################################
# default
################################################################################
@file = "mushroom.txt"
@n = 5
@m = 5
@k = 2
@@swap = false

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
  opts.on("-f [string]", "--file", "data file") { |f|
    @file = f
  }
  opts.on("-n [int]", "# folds of one cross-validation") { |f|
    @n = f.to_i
  }
  opts.on("-m [int]", "# cross-validations") { |f|
    @m = f.to_i
  }
  opts.on("-k [int]", "length of decision list ") { |f|
    @k = f.to_i
  }
  opts.on("-s", "--swap", "swap te & tr size"){
    @@swap = true
  }
  # parse
  opts.parse!(ARGV)
}

################################################################################
# data
################################################################################
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

    show
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
    puts "Attributes"
    puts "attr = #{@vals.size}"
    for i in 0..@vals.size-1
      puts "attr #{i} = #{@vals[i].size} values:  #{@vals[i].keys.join(", ")}"
    end

    # data
    puts "Data"
    puts "# samples = #{@data.size}"

    # class size
    c = @vals[0].size
    num = Array.new(c){|i| 0}
    @data.each do |datum|
      num[datum[0]] += 1
    end
    for i in 0..c-1
      puts "class #{i+1} = #{num[i]}"
    end
    @data[0..10].each do |datum|
      puts "#{datum}"
    end
    puts "..."
  end
end

################################################################################
# base learner
################################################################################
class MyLearner
  #### reset learner: arg = # values of each attributes ####
  def reset(prop)
    @c = prop.shift # # class
    @n = prop.size  # # attributes
    @v = prop       # # values of each attribute

    init
  end

  #### initialize ####
  def init
  end

  #### show ####
  def show
  end

  #### learn from data ####
  def learn(data)
    data.each do |datum|
      dc = datum[0]
      da = datum[1..@n]
      learn_i(dc, da)
    end
  end
  def learn_i(dc, da)
  end

  #### predict ####
  def predict(datum)
    return rand(@c)
  end
end

################################################################################
# tester
################################################################################
class MyTester
  #### new ####
  def initialize(file)
    @data = MyData.new(file)
    @sets = nil
  end

  #### test : m * n-ford cross validation ####
  def test(ls, n, m)
    puts "#{n}-fold cross validation * #{m} retry"
    puts "----------------------------------------"
    afs = Array.new(ls.size){|i| 0}
    # repeat cv m times
    for i in 1..m
      puts "Cross Validation #{i}"
      fs = cv(ls, n)
      for j in 0..ls.size-1
        afs[j] += fs[j]
      end
      puts "----------------------------------------"
    end
    afs.map!{|f| f /= m.to_f}
    puts "Ave #{afs.map{|af| sprintf("%10.5e", af)}.join(" ")}"

    afs
  end

  #### n-fold cross validation ####
  def cv(ls, n)
    @sets = @data.split(n)           # testing datasets
    afs = Array.new(ls.size){|i| 0}  # averaged f-measures of each learner

    for i in 0..n-1
      fs = fold(ls, i)
      printf("%3d %s\n", i+1, fs.map{|f| sprintf("%10.5e", f)}.join(" "))
      for j in 0..ls.size-1
        afs[j] += fs[j]
      end
    end
    afs.map!{|f| f /= n.to_f}
    puts "Ave #{afs.map{|af| sprintf("%10.5e", af)}.join(" ")}"

    afs
  end

  #### i-th fold ####
  def fold(ls, i)
    te = @sets[i]          # test set
    tr = @data.data - te   # training set
    if @@swap
      tmp = tr
      tr = te
      te = tmp
    end

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
        tc = key[0]        # true class
        pc = key[1]        # predicted class
        n = pred[key]
        tp += n if tc == c && pc == c
        tn += n if tc != c && pc != c
        fp += n if tc != c && pc == c
        fn += n if tc == c && pc != c
      end

      # recall, presision
      rec = tp / (tp + fn).to_f
      pre = tp / (tp + fp).to_f

      rec = 0 if tp + fn == 0
      pre = 0 if tp + fp == 0

      # f-measure
      f = 2 * rec * pre / (rec + pre)
      f = 0 if rec + pre == 0

      ave_f += f
    end
    ave_f /= @c.to_f
  end
end

################################################################################
# disj learner
################################################################################
class MyDisjLearner < MyLearner
  #### init ####
  def init
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
  def learn_i(dc, da)
    # remove da from disj[c != dc]
    for c in 0..@c-2
      next if c == dc
      for i in 0..@n-1
        @disj[c][i] -= [da[i]]
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
# Naive Bayes
################################################################################
class MyNaiveBayesLearner < MyLearner
  #### overwrite init ####
  def init
    @nc = Array.new(@c){|c| 0} # #{C = c}
    @nciv = Array.new(@c){|c|
      Array.new(@n){|i| Array.new(@v[i]){|v| 0}} # #{C=c, x_i = v}
    }
  end

  #### overwrite learn_i ####
  def learn_i(dc, da)
    @nc[dc] += 1
    for i in 0..@n-1
      @nciv[dc][i][da[i]] += 1
    end
  end

  #### overwirte predict ####
  def predict(da)
    # log p(da | c)
    lpc = Array.new(@c){|c| 0}
    for c in 0..@c-1
      for i in 0..@n-1
        lpc[c] += Math.log(@nciv[c][i][da[i]]) - Math.log(@nc[c])
      end
    end

    # log p(c)
    n = 0
    @nc.each{|c| n += c}
    lpc[c] += Math.log(@nc[c]) - Math.log(n) 

    # max_c = argmax_{c} log p(da | c) + log p(c)
    max_c = 0
    max_l = lpc[0]
    for c in 1..@c-1
      if max_l < lpc[c]
        max_c = c
        max_l = lpc[c]
      end
    end
    max_c
  end
end

################################################################################
# dl learner
################################################################################
class MyDLLearner < MyLearner
  #### new ####
  def initialize(k)
    @k = k
  end

  #### overwrite init ####
  def init
    @dl = [] # decision list
  end

  def show
    @dl.each do |t|
      p t
    end
  end

  #### overwrite learn ####
  def learn(data)
    # divide data by class
    ss = Array.new(@c){|i| []}
    data.each do |datum|
      dc = datum[0]
      da = datum[1..datum.size-1]
      ss[dc].push(da)
    end

    # learn decision list
    while eval(ss.map{|i| i.size}.join(" + ")) > 0
      # get a best term for each c
      best = get_best_term(ss)
      t = best[0]
      e = best[1]

      # remove explained samples
      for c in 0..@c-1
        ss[c] -= e[c]
      end

      # add t to decision list
      m = 0
      for c in 1..@c-1
        m = c if e[c].size > e[m].size
      end

      @dl.push([t, m])
    end
  end

  #### get the best term for s ####
  def get_best_term(ss)
    # unit clause
    cs = []  # candidats
    for i in 0..@n-1
      for v in 0..@v[i]-1
        l = [i, v]             # literal
        e = explained(l, ss)   # samples explained by l
        cs.push([[l], e])
      end
    end

    # find the best
    best = nil # best
    while (c = cs.shift) != nil
      t = c[0]
      e = c[1]
      best = c if best == nil || better?(e, best[1])
      next if t.size == @k

      # enumerate children
      l = t.last
      for i in l[0]+1..@n-1
        for v in 0..@v[i]-1
          cl = [i, v]
          ce = explained(cl, e)
          cs.push([t.clone + [cl], ce]) if eval(ce.map{|i| i.size}.join(" + ")) > 0
        end
      end
    end

    best
  end

  #### samples explained by a literal l (not a term!) ####
  def explained_la(l, a)
    a[ l[0] ] == l[1]
  end
  def explained_ta(t, a)
    t.each do |l|
      return false if !explained_la(l, a)
    end
    return true
  end
  def explained(l, ss)
    e = Array.new(@c){|i| [] }

    for c in 0..@c-1
      ss[c].each do |da|
        e[c].push(da) if explained_la(l, da)
      end
    end

    e
  end

  #### e1 is better than e2? ####
  def better?(e1, e2)
    ev1 = evaluate(e1)
    ev2 = evaluate(e2)
    ev1[0] < ev2[0] || (ev1[0] == ev2[0] && ev1[1] > ev2[1])
  end

  #### return [min, max] ####
  def evaluate(e)
    max = e[0].size
    for c in 1..@c-1
      max = e[c].size if max < e[c].size
    end

    sum = eval( e.map{|i| i.size}.join(" + ") )
    min = sum - max

    [min,  max]
  end

  #### overwrite predict ####
  def predict(da)
    @dl.each do |t, c|
      return c if explained_ta(t, da)
    end
    return 0 # by definition
  end
end

################################################################################
# main
################################################################################
# learners
ls = []
ls.push(MyLearner.new)                   # random
ls.push(MyDisjLearner.new)               # disj learner
ls.push(MyNaiveBayesLearner.new)         # naive Bayes
ls.push(MyDLLearner.new(@k))             # k-DL learner

# repeat n-fold closs validation m times
t = MyTester.new(@file)
t.test(ls, @n, @m)
