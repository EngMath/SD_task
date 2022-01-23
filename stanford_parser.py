

from jpype import startJVM, shutdownJVM, JPackage, getDefaultJVMPath
class stanfordParser:
    def __init__(self):
        #startJVM("/usr/lib/jvm/default-java/jre/lib/amd64/server/libjvm.so",
        #        "-Xms1024m", "-Xmx1024m", 
        #        "-Djava.class.path=/var/www/pygrammar/stanford-parser.jar:")
        #startJVM(getDefaultJVMPath(),"-Xms1024m", "-Xmx1024m", "-Djava.class.path=%s" % ('D:/stanford-parser-full-2017-06-09/stanford-parser.jar'))
        #startJVM(getDefaultJVMPath(), "-Djava.class.path=%s" % ('D:/pygrammer/stanford-parser.jar'))
        startJVM(getDefaultJVMPath(), "-Djava.class.path=%s" % ('parser/stanford-parser.jar'))
        self.Proc = JPackage('edu').stanford.nlp.process.DocumentPreprocessor();
        
        self.LexicalizedParser = JPackage('edu').stanford.nlp.parser.lexparser.LexicalizedParser
        self.Parser = self.LexicalizedParser('parser/englishPCFG.ser')
        
        self.Penn = JPackage('edu').stanford.nlp.trees.PennTreebankLanguagePack()
        self.Grammar = self.Penn.grammaticalStructureFactory()

    def tokenize( self, docu, result=[] ):
        t = self.Proc.getSentencesFromText( JPackage("java.io").StringReader( docu ) );
        result = [];
        for s in t:
            r = {};
            r["org_tokens"] = s;
            r["words"] = [str(w) for w in s];
            result.append(r);
        return result;

    def parse( self, r ):
        t = self.Parser.apply( r["org_tokens"] );
        r["org_tree"] = t;
        r["tags"] = [ str(n.parent(t).value()) for n in t.getLeaves()];
        gs = self.Grammar.newGrammaticalStructure( t )
        tdl = gs.typedDependenciesCollapsed()
        r["org_dependency"] = tdl;
        r["dep"] = [(str(dep.reln()), dep.gov().index(), dep.dep().index()) for dep in tdl];
        return r
    def __del__(self):

        shutdownJVM();


def get_offset(data_, tokenized):
    indices = [];
    begin = 0;
    for s in tokenized:
        ind = [];
        for w in s:
            begin = data_.find(w, begin);
            ind.append([begin, begin+len(w)]);
            begin += len(w);
        indices.append(ind);
    return indices;

