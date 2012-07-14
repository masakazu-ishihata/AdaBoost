#!/usr/bin/env ruby

################################################################################
# default
################################################################################
@file = "mushroom.txt"
@n = 5
@m = 5
@k = 2
@t = 100
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
  opts.on("-t [int]", "maximum number of iteration of AdaBoost") { |f|
    @t = f.to_i
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
  def initialize(ary)
    @c = ary[0]
    @a = ary[1..ary.size-1]
  end
  attr_reader :c, :a

  def show
    puts "#{@c} : #{a}"
  end
end
class MyDataset
  #### new ####
  def initialize(file)
    @file = file
    @size = 0
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

      @size += 1
      @data.push( MyData.new(datum) )
    end
  end

  #### accessors ####
  attr_reader :data, :size

  #### # values of each attributes ####
  def property
    @vals.map{|i| i.size}
  end

  #### split the dataset into n datasets ####
  def split(n)
    size = (@size / n.to_f).to_i # size of a sub-dataset
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
    ptn = Hash.new([])
    @data.each do |datum|
      num[ datum.c ] += 1
    end
    for i in 0..c-1
      puts "class #{i+1} = #{num[i]}"
    end
  end
end

################################################################################
# base learner
################################################################################
class MyLearner
  #### reset learner: arg = # values of each attributes ####
  def reset(prop)
    @c = prop[0]              # # class
    @n = prop.size-1          # # attributes
    @v = prop[1..prop.size-1] # # values of each attribute
    init
  end

  #### initialize ####
  def init
  end

  #### learn from data ####
  def learn(data)
    data.each do |datum|
      learn_i(datum.c, datum.a)
    end
  end
  def learn_i(dc, da)
  end

  #### predict ####
  def predict(data)
    pred = []

    # predict for each data
    data.each do |datum|
      pc = predict_i(datum.a)        # predicted class
      pred.push([datum.c, pc])
    end

    # f-measures
    ave_f = 0                # average f-measure for all c

    for c in 0..@c-1       # positive class
      # {true, false} * {positive, negative}
      tp = 0
      tn = 0
      fp = 0
      fn = 0
      pred.each do |tc, pc|
        tp += 1 if tc == c && pc == c
        tn += 1 if tc != c && pc != c
        fp += 1 if tc != c && pc == c
        fn += 1 if tc == c && pc != c
      end

      # recall, presision, f-measure
      rec = tp / (tp + fn).to_f
      pre = tp / (tp + fp).to_f

      rec = 0 if tp + fn == 0
      pre = 0 if tp + fp == 0

      f = 2 * rec * pre / (rec + pre) if (rec + pre) > 0
      f = 0 if rec + pre == 0

      ave_f += f
    end

    ave_f /= @c.to_f
  end
  def predict_i(da)
    rand(@c)
  end

  #### pac ####
  def pac(eps, del)
    -1
  end
end

################################################################################
# tester
################################################################################
class MyTester
  #### new ####
  def initialize(file)
    @dset = MyDataset.new(file)
    @sets = nil
  end

  #### test : m * n-ford cross validation ####
  def test(ls, n, m)
    puts "----------------------------------------"
    @dset.show
    puts "----------------------------------------"
    puts "#{n}-fold cross validation * #{m} retry"

    # PAC
    if @@swap
      puts "# samples s = #{(@dset.size / n.to_i).to_i}"
    else
      puts "# samples s = #{(@dset.size / n.to_i).to_i * (n-1)}"
    end

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
    @sets = @dset.split(n)           # testing datasets
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
    tr = @dset.data - te   # training set
    if @@swap
      tmp = tr
      tr = te
      te = tmp
    end

    fs = [] # averaged f-measures

    for i in 0..ls.size-1
      # reset
      ls[i].reset(@dset.property)

      # learn
      ls[i].learn(tr)

      # test
      f = ls[i].predict(te)
      fs.push(f)
    end

    fs
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
  def predict_i(da)
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
# disj learner
################################################################################
class MyDisjLearner < MyLearner
  #### init ####
  def init
    @disj = Array.new(@c-1){ |c|
      Array.new(@n){|i| Array.new(@v[i]){|v| v}}  # disj for class c
    }
  end

  #### learn ####
  def learn_i(dc, da)
    # remove da from disj[c != dc]
    for c in 0..@c-2
      next if c == dc
      for i in 0..@n-1
        @disj[c][i].delete(da[i])
      end
    end
  end

  #### predict ####
  def predict_i(da)
    for c in 0..@c-2
      for i in 0..@n-1
        return c if @disj[c][i].index(da[i]) != nil
      end
    end
    return @c-1
  end

  #### pac ####
  def pac(ep, de)
    (@n * Math.log(3) - Math.log(de)) / ep
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

  #### overwrite show ####
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
      ss[datum.c].push(datum.a)
    end

    # learn decision list
    while eval(ss.map{|i| i.size}.join(" + ")) > 0
      # get the best term
      best = get_best_term(ss)
      t = best[0]
      e = best[1]

      # remove explained samples by the best
      for c in 0..@c-1
        ss[c] -= e[c]
      end

      # add the best to decision list
      m = 0
      for c in 1..@c-1
        m = c if e[c].size > e[m].size
      end
      @dl.push([t, m])
    end
  end

  #### get the best term ####
  def get_best_term(ss)
    # generate unit cluases
    cs = []
    for i in 0..@n-1
      for v in 0..@v[i]-1
        l = [i, v]             # literal
        e = explained(l, ss)   # samples explained by l
        cs.push([[l], e])
      end
    end

    # search the best
    best = nil
    while (c = cs.shift) != nil
      t = c[0]
      e = c[1]

      best = c if best == nil || better?(e, best[1])
      next if t.size == @k

      # add children
      l = t.last
      for i in l[0]+1..@n-1
        for v in 0..@v[i]-1
          cl = [i, v]            # add a new literal
          ce = explained(cl, e)  # remove samples not explained by cl
          cs.push([t.clone + [cl], ce]) if eval(ce.map{|i| i.size}.join(" + ")) > 0
        end
      end
    end

    best
  end

  #### a is explained by a literal l ####
  def explained_la(l, a)
    a[ l[0] ] == l[1]
  end

  #### a is explained by a term t ####
  def explained_ta(t, a)
    t.each do |l|
      return false if !explained_la(l, a)
    end
    return true
  end

  #### samples explained by a literal ####
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
  # max : # samples in the target class
  # min : # samples in NOT target class
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
  def predict_i(da)
    @dl.each do |t, c|
      return c if explained_ta(t, da)
    end
    return 0 # by definition
  end

  #### pac ####
  def pac(ep, de)
    (4 * @n * @n * (6 + Math.log(@n)) - Math.log(de)) / ep
  end
end

################################################################################
# AdaBoost : weak hypothesis = single decision rule
################################################################################
class MyWeakLearner < MyLearner
  def initialize(i, v, c)
    @i = i
    @v = v
    @c = c
  end

  def predict_i(da)
    return @c  if da[@i] == @v
    return 1   if @c == 0
    return 0
  end
end

class MyAdaBoost < MyLearner
  def initialize(t)
    @t = t
  end

  #### overwrite init ####
  def init
    @hs = []     # hypotheses
    @as = []     # alphas : importance rate

    # weak learners (they learn NOTHING)
    @wl = []
    for i in 0..@n-1
      for v in 0..@v[i]-1
        for c in 0..@c-1
          @wl.push(MyWeakLearner.new(i, v, c))
        end
      end
    end
  end

  #### overwrite learn ####
  def learn(data)
    # initial weight
    m = data.size
    w = Array.new(m){|i| 1 / m.to_f}
    t = 0

    begin
      t += 1

      h = best_wl(data, w)                 # best hypothesis
      g = 0.5 - error(h, data, w)          # gamma : advantage
      b = Math.sqrt( (1-2*g) / (1+2*g) )   # beta
      a = -Math.log(b)                     # alpha : importance rate

      @hs.push(h)
      @as.push(a)

      # udate weights
      e = 0
      for i in 0..data.size-1
        pc = predict_i(data[i].a)
        w[i] *= b if data[i].c == pc
        w[i] /= b if data[i].c != pc
        e += 1    if data[i].c != pc
      end
    end while e > 0 && t < @t
  end

  #### choose a best weak learner (hypothesis) ####
  def best_wl(data, weight)
    best = nil
    min = 0

    @wl.each do |wl|
      e = error(wl, data, weight)
      if best == nil || min > e
        best = wl
        min = e
      end
    end

    best
  end

  #### error ####
  def error(wl, data, weight)
    e = 0
    for i in 0..data.size-1
      e += weight[i] if data[i].c != wl.predict_i(data[i].a)
    end
    e
  end

  #### overwrite predict ####
  def predict_i(da)
    vote = Array.new(@c){|i| 0}
    for t in 0..@hs.size-1
      pc = @hs[t].predict_i(da)
      vote[pc] += @as[t]
    end

    m   = 0
    max = vote[0]
    for c in 1..@c-1
      if max < vote[c]
        m   = c
        max = vote[c]
      end
    end

    m
  end
end

################################################################################
# main
################################################################################
# learners
ls = []
ls.push(MyLearner.new)                   # random
ls.push(MyNaiveBayesLearner.new)         # naive Bayes
ls.push(MyDisjLearner.new)               # disj learner
ls.push(MyDLLearner.new(@k))             # k-DL learner
ls.push(MyAdaBoost.new(@t))              # AdaBoost with maximum iteration t

puts "learner 1 : random"
puts "learner 2 : naive Bayes"
puts "learner 3 : Disj learner"
puts "learner 4 : k-DL learner"
puts "learner 5 : AdaBoost"

# repeat n-fold closs validation m times
t = MyTester.new(@file)
t.test(ls, @n, @m)
